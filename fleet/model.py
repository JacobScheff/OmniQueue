"""Pointer actor-critic for fleet dispatch.

Vehicles are coordinated with a transformer (guests → vehicles). Actions are a
masked pointer over padded request slots plus STAY/IDLE — not a softmax over
intersections. Optional mean-aggregation GNN encodes the street graph into
fixed-width node embeddings used for vehicle/request locations.
"""

import math

import torch
import torch.nn as nn

import fleet.config as config


# Graph Transformer (intersection + street network)
class GraphTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = config.D_MODEL
        n_heads = config.NUM_ATTN_HEADS
        assert d_model % n_heads == 0

        self.num_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_spd = config.MAX_SHORTEST_PATH_DIST

        self.node_feature_expansion = nn.Linear(config.INTERSECTION_DYNAMIC_FEAT_DIM, d_model)
        self.laplacian_pe_proj = nn.Linear(config.LAPLACIAN_PE_DIM, d_model)

        # Full attention (Q/K/V) + output projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # B_φ(i,j): learned per-head bias indexed by shortest-path hop distance
        self.spatial_bias = nn.Embedding(self.max_spd + 1, n_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _shortest_path_hops(self, A: torch.Tensor) -> torch.Tensor:
        """All-pairs hop distances via Floyd–Warshall; unreachable clamped to max_spd."""
        n = A.size(-1)
        unreachable = self.max_spd + 1
        dist = torch.full(A.shape, unreachable, device=A.device, dtype=torch.long)
        dist = dist.masked_fill(A > 0, 1)
        idx = torch.arange(n, device=A.device)
        dist[..., idx, idx] = 0

        for k in range(n):
            via = dist[..., :, k].unsqueeze(-1) + dist[..., k, :].unsqueeze(-2)
            dist = torch.minimum(dist, via)
        return dist.clamp(max=self.max_spd)

    def _attend(self, x: torch.Tensor, streets: torch.Tensor) -> torch.Tensor:
        """All-to-all attention with shortest-path and edge biases."""
        *batch, n, _ = x.shape
        h, d_h = self.num_heads, self.d_head

        q = self.q_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)
        k = self.k_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)
        v = self.v_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)

        # QK^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)

        # + B_φ(i,j) from shortest-path distance
        spd = self._shortest_path_hops(streets)
        scores = scores + self.spatial_bias(spd).movedim(-1, -3)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.movedim(-3, -2).contiguous().view(*batch, n, h * d_h)
        return self.out_proj(out)

    def forward(self, intersections, streets):
        # Expand intersection features to d_model
        intersections = self.node_feature_expansion(intersections)

        # --- Structural Encodings --- #

        # Graph Laplacian = D - A, where D is the degree matrix and A is the adjacency matrix
        A = streets
        D = torch.diag_embed(A.sum(dim=-1))
        L = D - A

        # Compute the Laplacian positional encodings (top-k eigenvectors)
        # eigenvectors: [..., n, k] — row i is node i's k-dim structural coordinates
        _, eigenvectors = torch.lobpcg(L, k=config.LAPLACIAN_PE_DIM, largest=True)

        # Assign coordinates: SE_i = (u_1[i], ..., u_k[i]); project to d_model and inject
        structural_encoding = self.laplacian_pe_proj(eigenvectors)
        x = intersections + structural_encoding

        # --- Full Attention with Edge Biases --- #
        # Attn(i,j) = Softmax(Q_i K_j^T / sqrt(d) + B_φ(i,j) + edge_bias) V_j
        x = self.norm1(x + self._attend(x, streets))

        # --- FFN (shared across nodes) + second residual / LayerNorm --- #
        x = self.norm2(x + self.ffn(x))
        return x

# Vehicle ↔ vehicle self-attention (wave coordinator)
class VehicleCoordinatorLayer(nn.Module):
    """One self-attention + FFN block over the free-vehicle axis."""

    def __init__(self, d_model: int, num_heads: int, d_head: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model, num_heads * d_head)
        self.k_proj = nn.Linear(d_model, num_heads * d_head)
        self.v_proj = nn.Linear(d_model, num_heads * d_head)
        self.out_proj = nn.Linear(num_heads * d_head, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _attend(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Self-attention over vehicles.

        key_padding_mask: (..., n) True = valid vehicle (False = pad).
        """
        *batch, n, _ = x.shape
        h, d_h = self.num_heads, self.d_head

        q = self.q_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)
        k = self.k_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)
        v = self.v_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)

        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask.unsqueeze(-2).unsqueeze(-2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v)
        out = out.movedim(-3, -2).contiguous().view(*batch, n, h * d_h)
        return self.out_proj(out)

    def forward(
        self,
        vehicles: torch.Tensor,
        vehicle_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(vehicles + self._attend(vehicles, vehicle_padding_mask))
        x = self.norm2(x + self.ffn(x))
        return x

# Vehicle cross Request Attention
class VehicleRequestCrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model, num_heads * d_head)
        self.k_proj = nn.Linear(d_model, num_heads * d_head)
        self.v_proj = nn.Linear(d_model, num_heads * d_head)
        self.out_proj = nn.Linear(num_heads * d_head, d_model)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention: queries attend over keys/values.

        key_padding_mask: (..., n_kv) True = valid key (False = pad).
        """
        *batch, n_q, _ = q.shape
        n_kv = k.shape[-2]
        h, d_h = self.num_heads, self.d_head

        q = self.q_proj(q).view(*batch, n_q, h, d_h).movedim(-2, -3)
        k = self.k_proj(k).view(*batch, n_kv, h, d_h).movedim(-2, -3)
        v = self.v_proj(v).view(*batch, n_kv, h, d_h).movedim(-2, -3)

        # QK^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)

        if key_padding_mask is not None:
            # Broadcast over heads: (..., 1, 1, n_kv)
            scores = scores.masked_fill(~key_padding_mask.unsqueeze(-2).unsqueeze(-2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.movedim(-3, -2).contiguous().view(*batch, n_q, h * d_h)
        return self.out_proj(out)

    def forward(
        self,
        vehicle: torch.Tensor,
        requests: torch.Tensor,
        request_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._attend(vehicle, requests, requests, key_padding_mask=request_padding_mask)