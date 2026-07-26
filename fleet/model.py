"""Pointer actor-critic for fleet dispatch.

Vehicles are coordinated with a transformer (guests → vehicles). Actions are a
masked pointer over padded request slots plus STAY/IDLE — not a softmax over
intersections. Optional Graph Transformer encodes the street graph into
fixed-width node embeddings used for vehicle/request locations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import fleet.config as config


def _shortest_path_hops(A: torch.Tensor, max_spd: int) -> torch.Tensor:
    """All-pairs unweighted hop distances via Floyd–Warshall.

    Uses connectivity (A > 0), ignoring edge weights. Unreachable pairs are
    clamped to ``max_spd`` so they share the farthest spatial-bias bucket.
    """
    n = A.size(-1)
    unreachable = max_spd + 1
    dist = torch.full(A.shape, unreachable, device=A.device, dtype=torch.long)
    dist = dist.masked_fill(A > 0, 1)
    idx = torch.arange(n, device=A.device)
    dist[..., idx, idx] = 0

    for k in range(n):
        via = dist[..., :, k].unsqueeze(-1) + dist[..., k, :].unsqueeze(-2)
        dist = torch.minimum(dist, via)
    return dist.clamp(max=max_spd)


def _laplacian_pe(A: torch.Tensor, k: int, training: bool) -> torch.Tensor:
    """Smallest non-trivial Laplacian eigenvectors; shape (..., n, k).

    Uses a symmetrized adjacency so ``eigh`` is well-defined on directed street
    graphs. Skips the trivial (≈0) eigenmode. Randomly flips eigenvector signs
    during training (standard LapPE augmentation).
    """
    # Symmetrize for an undirected combinatorial Laplacian.
    A_sym = torch.maximum(A, A.transpose(-2, -1)).to(dtype=torch.float32)
    deg = A_sym.sum(dim=-1)
    L = torch.diag_embed(deg) - A_sym

    # Batched eigh: eigenvalues ascending, columns = eigenvectors.
    _, evecs = torch.linalg.eigh(L)
    n = A_sym.size(-1)
    take = min(k, max(n - 1, 0))

    pe = evecs.new_zeros(*evecs.shape[:-1], k)
    if take > 0:
        pe[..., :, :take] = evecs[..., :, 1 : 1 + take]
        if training:
            signs = (
                torch.empty(
                    *pe.shape[:-2],
                    1,
                    k,
                    device=pe.device,
                    dtype=pe.dtype,
                )
                .bernoulli_(0.5)
                .mul_(2)
                .sub_(1)
            )
            pe = pe * signs
    return pe


class GraphTransformerLayer(nn.Module):
    """One full-attention + FFN block over graph nodes with spatial/edge biases."""

    def __init__(self, d_model: int | None = None, n_heads: int | None = None):
        super().__init__()
        d_model = d_model if d_model is not None else config.D_MODEL
        n_heads = n_heads if n_heads is not None else config.NUM_ATTN_HEADS
        assert d_model % n_heads == 0

        self.num_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_spd = config.MAX_SHORTEST_PATH_DIST

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # B_φ(i,j): learned per-head bias indexed by shortest-path hop distance
        self.spatial_bias = nn.Embedding(self.max_spd + 1, n_heads)
        # Binary edge presence bias (0 = no edge, 1 = edge)
        self.edge_bias = nn.Embedding(2, n_heads)

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
        spd: torch.Tensor,
        streets: torch.Tensor,
    ) -> torch.Tensor:
        """Attn(i,j) = Softmax(Q_i K_j^T / √d + B_φ(i,j) + edge_bias) V_j."""
        *batch, n, _ = x.shape
        h, d_h = self.num_heads, self.d_head

        q = self.q_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)
        k = self.k_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)
        v = self.v_proj(x).view(*batch, n, h, d_h).movedim(-2, -3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        scores = scores + self.spatial_bias(spd).movedim(-1, -3)

        edge = (streets > 0).long()
        scores = scores + self.edge_bias(edge).movedim(-1, -3)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.movedim(-3, -2).contiguous().view(*batch, n, h * d_h)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        spd: torch.Tensor,
        streets: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self._attend(x, spd, streets))
        x = self.norm2(x + self.ffn(x))
        return x


class GraphEncoder(nn.Module):
    """Street-graph encoder: LapPE once, SPD once, then ``NUM_GNN_LAYERS`` blocks."""

    def __init__(
        self,
        d_model: int | None = None,
        n_heads: int | None = None,
        num_layers: int | None = None,
    ):
        super().__init__()
        d_model = d_model if d_model is not None else config.D_MODEL
        n_heads = n_heads if n_heads is not None else config.NUM_ATTN_HEADS
        num_layers = num_layers if num_layers is not None else config.NUM_GNN_LAYERS

        self.max_spd = config.MAX_SHORTEST_PATH_DIST
        self.pe_dim = config.LAPLACIAN_PE_DIM

        self.node_feature_expansion = nn.Linear(
            config.INTERSECTION_DYNAMIC_FEAT_DIM, d_model
        )
        self.laplacian_pe_proj = nn.Linear(self.pe_dim, d_model)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d_model, n_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        intersections: torch.Tensor,
        streets: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_feature_expansion(intersections)

        pe = _laplacian_pe(streets, self.pe_dim, training=self.training)
        pe = pe.to(dtype=x.dtype)
        x = x + self.laplacian_pe_proj(pe)

        # Structural biases shared across all stacked layers.
        spd = _shortest_path_hops(streets, self.max_spd)
        for layer in self.layers:
            x = layer(x, spd, streets)
        return x


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
            scores = scores.masked_fill(
                ~key_padding_mask.unsqueeze(-2).unsqueeze(-2), float("-inf")
            )

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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                ~key_padding_mask.unsqueeze(-2).unsqueeze(-2), float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)
        # All-padded keys → softmax(NaN); zero those rows out.
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v)
        out = out.movedim(-3, -2).contiguous().view(*batch, n_q, h * d_h)
        return self.out_proj(out)

    def forward(
        self,
        vehicle: torch.Tensor,
        requests: torch.Tensor,
        request_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._attend(
            vehicle, requests, requests, key_padding_mask=request_padding_mask
        )


class VehicleRouter(nn.Module):
    """Masked pointer actor-critic over request slots + STAY/IDLE.

    Park-style stack: encode vehicles (decision agents) and requests (candidates),
    coordinate free vehicles with a transformer, optionally fuse via cross-attention,
    then score with a pointer head ``q·k/√d`` over padded request slots.
    """

    def __init__(
        self,
        vehicle_feat_dim: int = config.VEHICLE_DYNAMIC_FEAT_DIM,
        request_feat_dim: int = config.REQUEST_DYNAMIC_FEAT_DIM,
        pairwise_feat_dim: int = config.PAIRWISE_DYNAMIC_FEAT_DIM,
        environment_feat_dim: int = config.ENV_DYNAMIC_FEAT_DIM,
        d_model: int = config.D_MODEL,
        num_actions: int | None = None,
        num_layers: int = config.NUM_TRANSFORMER_LAYERS,
        num_heads: int = config.NUM_ATTN_HEADS,
        max_requests: int = config.MAX_REQUESTS,
        max_nodes: int = config.MAX_NODES,
        use_graph_encoder: bool = True,
        num_gnn_layers: int = config.NUM_GNN_LAYERS,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )

        self.max_requests = max_requests
        self.num_actions = num_actions or (max_requests + config.NUM_SPECIAL_ACTIONS)
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_graph_encoder = use_graph_encoder
        self.environment_feat_dim = environment_feat_dim

        d_head = d_model // num_heads

        # Location encodings: graph transformer and/or node-id fallback.
        self.graph_encoder = (
            GraphEncoder(d_model, num_heads, num_gnn_layers)
            if use_graph_encoder
            else None
        )
        self.node_id_embed = nn.Embedding(max_nodes, d_model)

        self.vehicle_proj = nn.Sequential(
            nn.Linear(vehicle_feat_dim + environment_feat_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.vehicle_norm = nn.LayerNorm(d_model)

        # Shared request encoder (dynamics + origin/dest). Pairwise is added per
        # vehicle only on the pointer key path.
        self.request_feat_proj = nn.Sequential(
            nn.Linear(request_feat_dim + 2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.request_norm = nn.LayerNorm(d_model)
        self.pairwise_proj = nn.Sequential(
            nn.Linear(pairwise_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pointer_key_norm = nn.LayerNorm(d_model)

        self.coordinator_layers = nn.ModuleList(
            [
                VehicleCoordinatorLayer(d_model, num_heads, d_head)
                for _ in range(num_layers)
            ]
        )

        self.cross_attn = VehicleRequestCrossAttentionLayer(d_model, num_heads, d_head)
        self.cross_norm = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.stay_idle_head = nn.Linear(d_model, config.NUM_SPECIAL_ACTIONS)

        self.critic_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + environment_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _node_embeddings(
        self,
        intersections: torch.Tensor | None,
        streets: torch.Tensor | None,
        num_nodes_hint: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return node embeddings (N, D) or (B, N, D)."""
        if (
            self.use_graph_encoder
            and self.graph_encoder is not None
            and intersections is not None
            and streets is not None
        ):
            return self.graph_encoder(intersections, streets)

        if intersections is not None and intersections.dim() == 3:
            batch, num_nodes, _ = intersections.shape
            ids = torch.arange(num_nodes, device=device)
            return self.node_id_embed(ids).to(dtype=dtype).unsqueeze(0).expand(
                batch, -1, -1
            )

        ids = torch.arange(num_nodes_hint, device=device)
        return self.node_id_embed(ids).to(dtype=dtype)

    def _gather_nodes(
        self,
        node_emb: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather node embeddings by index.

        node_emb: (N, D) or (B, N, D)
        index: (B, …) long
        returns: (B, …, D)
        """
        if node_emb.dim() == 2:
            return node_emb[index]
        batch = index.size(0)
        flat_index = index.reshape(batch, -1)
        batch_ids = torch.arange(batch, device=index.device).unsqueeze(-1)
        gathered = node_emb[batch_ids, flat_index]
        return gathered.view(*index.shape, node_emb.size(-1))

    def forward(
        self,
        vehicle_features: torch.Tensor,
        request_features: torch.Tensor,
        pairwise_features: torch.Tensor,
        environment_features: torch.Tensor,
        vehicle_padding_mask: torch.Tensor | None = None,
        request_padding_mask: torch.Tensor | None = None,
        vehicle_node_index: torch.Tensor | None = None,
        request_origin_index: torch.Tensor | None = None,
        request_dest_index: torch.Tensor | None = None,
        intersections: torch.Tensor | None = None,
        streets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        vehicle_features: (B, V, F_v)
        request_features: (B, R, F_r) or (B, V, R, F_r)
        pairwise_features: (B, V, R, F_p)
        environment_features: (B, F_e)
        vehicle_padding_mask: (B, V) True = valid vehicle
        request_padding_mask: (B, R) True = valid request slot
        vehicle_node_index / request_origin_index / request_dest_index: long indices
        intersections / streets: optional street graph for GraphEncoder

        Returns:
            logits: (B, V, num_actions) — request slots then STAY, IDLE
            values: (B, V, 1)
        """
        if pairwise_features.dim() == 3:
            # Unbatched (V, R, P) → add batch dim consistently.
            pairwise_features = pairwise_features.unsqueeze(0)
            vehicle_features = vehicle_features.unsqueeze(0)
            if request_features.dim() == 2:
                request_features = request_features.unsqueeze(0)
            environment_features = environment_features.unsqueeze(0)
            if vehicle_padding_mask is not None and vehicle_padding_mask.dim() == 1:
                vehicle_padding_mask = vehicle_padding_mask.unsqueeze(0)
            if request_padding_mask is not None and request_padding_mask.dim() == 1:
                request_padding_mask = request_padding_mask.unsqueeze(0)

        batch_size, num_vehicles, num_requests, _ = pairwise_features.shape
        if num_requests > self.max_requests:
            raise ValueError(
                f"num_requests={num_requests} exceeds max_requests={self.max_requests}"
            )

        if request_features.dim() == 4:
            # (B, V, R, F) → shared (B, R, F); dynamics are not vehicle-specific.
            request_features = request_features[:, 0]

        device = vehicle_features.device
        dtype = vehicle_features.dtype
        num_nodes_hint = config.MAX_NODES
        if intersections is not None:
            num_nodes_hint = intersections.size(-2)
        else:
            # Node-id fallback must cover every index we gather (vehicle +
            # request o/d). Taking only vehicle max under-sizes the table.
            maxima: list[int] = []
            for idx in (
                vehicle_node_index,
                request_origin_index,
                request_dest_index,
            ):
                if idx is not None and idx.numel() > 0:
                    maxima.append(int(idx.max().item()) + 1)
            if maxima:
                num_nodes_hint = max(maxima)

        node_emb = self._node_embeddings(
            intersections, streets, num_nodes_hint, device, dtype
        )

        if vehicle_node_index is None:
            veh_loc = torch.zeros(
                batch_size, num_vehicles, self.d_model, device=device, dtype=dtype
            )
        else:
            veh_loc = self._gather_nodes(node_emb, vehicle_node_index)

        if request_origin_index is None:
            req_o = torch.zeros(
                batch_size, num_requests, self.d_model, device=device, dtype=dtype
            )
        else:
            req_o = self._gather_nodes(node_emb, request_origin_index)

        if request_dest_index is None:
            req_d = torch.zeros(
                batch_size, num_requests, self.d_model, device=device, dtype=dtype
            )
        else:
            req_d = self._gather_nodes(node_emb, request_dest_index)

        env_for_vehicles = environment_features.unsqueeze(1).expand(-1, num_vehicles, -1)
        vehicle_inputs = torch.cat(
            [vehicle_features, env_for_vehicles, veh_loc], dim=-1
        )
        vehicle_embeddings = self.vehicle_norm(self.vehicle_proj(vehicle_inputs))

        coordinated = vehicle_embeddings
        for layer in self.coordinator_layers:
            coordinated = layer(coordinated, vehicle_padding_mask=vehicle_padding_mask)

        request_inputs = torch.cat([request_features, req_o, req_d], dim=-1)
        request_embeddings = self.request_norm(self.request_feat_proj(request_inputs))

        if request_padding_mask is not None:
            request_embeddings = request_embeddings * request_padding_mask.unsqueeze(
                -1
            ).to(dtype=request_embeddings.dtype)

        fused = coordinated + self.cross_attn(
            coordinated,
            request_embeddings,
            request_padding_mask=request_padding_mask,
        )
        coordinated = self.cross_norm(fused)

        # Pointer keys: shared request emb + per-vehicle pairwise.
        pairwise_emb = self.pairwise_proj(pairwise_features)  # (B, V, R, D)
        pointer_keys = self.pointer_key_norm(
            request_embeddings.unsqueeze(1) + pairwise_emb
        )

        queries = self.q_proj(coordinated)  # (B, V, D)
        keys = self.k_proj(pointer_keys)  # (B, V, R, D)
        # Per-vehicle pointer scores over request slots: (B, V, R)
        attention_scores = torch.einsum("bvd,bvrd->bvr", queries, keys) / (
            self.d_model ** 0.5
        )

        # Mask invalid / padded request slots so they cannot be sampled.
        if request_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~request_padding_mask.unsqueeze(1), float("-inf")
            )

        # Pad request logit axis to max_requests so action indices stay stable.
        if num_requests < self.max_requests:
            pad = attention_scores.new_full(
                (batch_size, num_vehicles, self.max_requests - num_requests),
                float("-inf"),
            )
            attention_scores = torch.cat([attention_scores, pad], dim=-1)

        stay_idle = self.stay_idle_head(coordinated)  # (B, V, 2)
        logits = torch.cat([attention_scores, stay_idle], dim=-1)

        if request_padding_mask is not None:
            denom = request_padding_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            avg_request = request_embeddings.sum(dim=1) / denom
            avg_request = avg_request.unsqueeze(1).expand(-1, num_vehicles, -1)
        else:
            avg_request = (
                request_embeddings.mean(dim=1)
                .unsqueeze(1)
                .expand(-1, num_vehicles, -1)
            )

        critic_input = torch.cat([coordinated, avg_request, env_for_vehicles], dim=-1)
        values = self.critic_mlp(critic_input)

        if vehicle_padding_mask is not None:
            values = values * vehicle_padding_mask.unsqueeze(-1).to(dtype=values.dtype)

        return logits, values


def _flat_obs_slices() -> dict[str, slice]:
    """Slice map for FleetEnv flat observations (must match FleetEnv.hpp)."""
    cursor = 0

    def take(n: int) -> slice:
        nonlocal cursor
        start = cursor
        cursor += n
        return slice(start, cursor)

    return {
        "vehicle": take(config.VEHICLE_DYNAMIC_FEAT_DIM),
        "request": take(config.MAX_REQUESTS * config.REQUEST_DYNAMIC_FEAT_DIM),
        "pairwise": take(config.MAX_REQUESTS * config.PAIRWISE_DYNAMIC_FEAT_DIM),
        "env": take(config.ENV_DYNAMIC_FEAT_DIM),
        "request_mask": take(config.MAX_REQUESTS),
        "action_mask": take(config.NUM_ACTIONS),
        "vehicle_node": take(1),
        "request_origin": take(config.MAX_REQUESTS),
        "request_dest": take(config.MAX_REQUESTS),
    }


_FLAT = _flat_obs_slices()


def _obs_flat_to_tensors(obs_flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unpack FleetEnv flat obs → VehicleRouter kwargs (batch of V=1)."""
    if obs_flat.dim() == 1:
        obs_flat = obs_flat.unsqueeze(0)
    b = obs_flat.shape[0]
    r = config.MAX_REQUESTS

    return {
        "vehicle_features": obs_flat[:, _FLAT["vehicle"]].view(
            b, 1, config.VEHICLE_DYNAMIC_FEAT_DIM
        ),
        "request_features": obs_flat[:, _FLAT["request"]].view(
            b, r, config.REQUEST_DYNAMIC_FEAT_DIM
        ),
        "pairwise_features": obs_flat[:, _FLAT["pairwise"]].view(
            b, 1, r, config.PAIRWISE_DYNAMIC_FEAT_DIM
        ),
        "environment_features": obs_flat[:, _FLAT["env"]],
        "vehicle_padding_mask": torch.ones(
            b, 1, dtype=torch.bool, device=obs_flat.device
        ),
        "request_padding_mask": obs_flat[:, _FLAT["request_mask"]] > 0.5,
        "vehicle_node_index": obs_flat[:, _FLAT["vehicle_node"]].long().view(b, 1),
        "request_origin_index": obs_flat[:, _FLAT["request_origin"]].long(),
        "request_dest_index": obs_flat[:, _FLAT["request_dest"]].long(),
        "action_mask": (obs_flat[:, _FLAT["action_mask"]] > 0.5).unsqueeze(1),
    }


def forward_with_mask(
    model: VehicleRouter,
    obs_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run VehicleRouter on flat FleetEnv obs with action masking.

    Returns masked logits (B, V, A) and values (B, V, 1).
    """
    tensors = _obs_flat_to_tensors(obs_flat)
    action_mask = tensors.pop("action_mask")
    logits, values = model(**tensors)
    # Model already masks padded request slots; also apply C++ legality mask.
    logits = logits.masked_fill(~action_mask, -1e9)
    return logits, values