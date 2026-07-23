"""Pointer actor-critic for fleet dispatch.

Vehicles are coordinated with a transformer (guests → vehicles). Actions are a
masked pointer over padded request slots plus STAY/IDLE — not a softmax over
intersections. Optional mean-aggregation GNN encodes the street graph into
fixed-width node embeddings used for vehicle/request locations.
"""

import torch
import torch.nn as nn

from fleet.config import (
    ACTION_IDLE,
    ACTION_STAY,
    D_MODEL,
    EDGE_FEAT_DIM,
    ENV_FEAT_DIM,
    MAX_NODES,
    MAX_REQUESTS,
    NODE_FEAT_DIM,
    NUM_ACTIONS,
    NUM_ATTN_HEADS,
    NUM_GNN_LAYERS,
    NUM_SPECIAL_ACTIONS,
    NUM_TRANSFORMER_LAYERS,
    PAIRWISE_FEAT_DIM,
    REQUEST_FEAT_DIM,
    USE_GNN,
    VEHICLE_FEAT_DIM,
    apply_action_mask,
    build_action_mask,
)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Mean-aggregate rows of ``src`` into ``dim_size`` bins by ``index``."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, idx, src)
    ones = torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index.unsqueeze(-1), ones)
    return out / count.clamp(min=1.0)


class MeanGNNLayer(nn.Module):
    """One GraphSAGE-style layer: mean neighbor messages (edge-aware) → MLP."""

    def __init__(self, d_model: int, edge_feat_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(d_model + edge_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # x: (N, D), edge_index: (2, M), edge_attr: (M, E)
        src, dst = edge_index[0], edge_index[1]
        msg = self.message(torch.cat([x[src], edge_attr], dim=-1))
        agg = scatter_mean(msg, dst, dim_size=x.size(0))
        return self.norm(self.update(torch.cat([x, agg], dim=-1)))


class GraphEncoder(nn.Module):
    """Fixed-width GNN over an arbitrary street graph."""

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        d_model: int = D_MODEL,
        num_layers: int = NUM_GNN_LAYERS,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [MeanGNNLayer(d_model, edge_feat_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode nodes.

        node_features: (N, F_n) or (B, N, F_n)
        edge_index: (2, M) — shared across batch when node_features is batched
        edge_attr: (M, F_e)
        returns: same leading shape as node_features with width d_model
        """
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        if node_features.dim() == 2:
            h = self.input_norm(self.input_proj(node_features))
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            return h

        # Batched independent graphs with a shared topology (same edge_index).
        batch, num_nodes, _ = node_features.shape
        outs = []
        for b in range(batch):
            h = self.input_norm(self.input_proj(node_features[b]))
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            outs.append(h)
        return torch.stack(outs, dim=0)


class VehicleTransformerBlock(nn.Module):
    """Self-attention + FFN over the free-vehicle axis (coordinator stack)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # key_padding_mask: (B, K) True = PAD (ignored), matching nn.MultiheadAttention
        attn_out, _ = self.attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class FleetRouterModel(nn.Module):
    """Masked pointer actor-critic over request candidates + STAY/IDLE."""

    def __init__(
        self,
        vehicle_feat_dim: int = VEHICLE_FEAT_DIM,
        request_feat_dim: int = REQUEST_FEAT_DIM,
        pairwise_feat_dim: int = PAIRWISE_FEAT_DIM,
        environment_feat_dim: int = ENV_FEAT_DIM,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        max_requests: int = MAX_REQUESTS,
        d_model: int = D_MODEL,
        num_actions: int | None = None,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        num_heads: int = NUM_ATTN_HEADS,
        num_gnn_layers: int = NUM_GNN_LAYERS,
        use_gnn: bool = USE_GNN,
        max_nodes: int = MAX_NODES,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.max_requests = max_requests
        self.num_actions = num_actions or (max_requests + NUM_SPECIAL_ACTIONS)
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_gnn = use_gnn
        self.vehicle_feat_dim = vehicle_feat_dim
        self.request_feat_dim = request_feat_dim
        self.pairwise_feat_dim = pairwise_feat_dim
        self.environment_feat_dim = environment_feat_dim

        self.graph_encoder: GraphEncoder | None
        self.node_id_embed: nn.Embedding | None
        if use_gnn:
            self.graph_encoder = GraphEncoder(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                d_model=d_model,
                num_layers=num_gnn_layers,
            )
            self.node_id_embed = None
        else:
            self.graph_encoder = None
            self.node_id_embed = nn.Embedding(max_nodes, d_model)

        self.vehicle_proj = nn.Sequential(
            nn.Linear(vehicle_feat_dim + environment_feat_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.vehicle_norm = nn.LayerNorm(d_model)

        self.coordinator_layers = nn.ModuleList(
            [VehicleTransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )

        # Request keys: shared request dynamics + per-vehicle pairwise + o/d node embs.
        req_in = request_feat_dim + pairwise_feat_dim + 2 * d_model
        self.request_feat_proj = nn.Sequential(
            nn.Linear(req_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.request_norm = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.special_head = nn.Linear(d_model, NUM_SPECIAL_ACTIONS)

        self.critic_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + environment_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _node_embeddings(
        self,
        node_features: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
        num_nodes_hint: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return node embeddings (N, D) or (B, N, D)."""
        if self.use_gnn:
            if (
                self.graph_encoder is None
                or node_features is None
                or edge_index is None
                or edge_attr is None
            ):
                raise ValueError(
                    "use_gnn=True requires node_features, edge_index, and edge_attr"
                )
            return self.graph_encoder(node_features, edge_index, edge_attr)

        assert self.node_id_embed is not None
        if node_features is not None and node_features.dim() == 3:
            batch, num_nodes, _ = node_features.shape
            ids = torch.arange(num_nodes, device=device)
            return self.node_id_embed(ids).unsqueeze(0).expand(batch, -1, -1)
        ids = torch.arange(num_nodes_hint, device=device)
        return self.node_id_embed(ids)

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
        node_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        vehicle_features: (B, K, V)
        request_features: (B, R, F_r) or (B, K, R, F_r)
        pairwise_features: (B, K, R, P)
        environment_features: (B, E)
        vehicle_padding_mask: (B, K) True = valid vehicle
        request_padding_mask: (B, R) True = valid request slot
        vehicle_node_index / request_origin_index / request_dest_index: long indices
        node_features / edge_index / edge_attr: optional street graph for GNN

        Returns:
            logits: (B, K, num_actions)  — request slots then STAY, IDLE
            values: (B, K, 1)
        """
        if pairwise_features.dim() == 3:
            pairwise_features = pairwise_features.unsqueeze(0)
            vehicle_features = vehicle_features.unsqueeze(0)
            if request_features.dim() == 2:
                request_features = request_features.unsqueeze(0)
            environment_features = environment_features.unsqueeze(0)

        batch_size, num_vehicles, num_requests, _ = pairwise_features.shape
        if num_requests > self.max_requests:
            raise ValueError(
                f"num_requests={num_requests} exceeds max_requests={self.max_requests}"
            )

        if request_features.dim() == 3:
            # (B, R, F) → (B, K, R, F)
            request_features = request_features.unsqueeze(1).expand(
                -1, num_vehicles, -1, -1
            )

        device = vehicle_features.device
        num_nodes_hint = MAX_NODES
        if node_features is not None:
            num_nodes_hint = node_features.size(-2)
        elif vehicle_node_index is not None:
            num_nodes_hint = int(vehicle_node_index.max().item()) + 1

        node_emb = self._node_embeddings(
            node_features, edge_index, edge_attr, num_nodes_hint, device
        )

        if vehicle_node_index is None:
            veh_loc = torch.zeros(
                batch_size, num_vehicles, self.d_model, device=device, dtype=vehicle_features.dtype
            )
        else:
            veh_loc = self._gather_nodes(node_emb, vehicle_node_index)

        if request_origin_index is None:
            req_o = torch.zeros(
                batch_size,
                num_requests,
                self.d_model,
                device=device,
                dtype=vehicle_features.dtype,
            )
        else:
            req_o = self._gather_nodes(node_emb, request_origin_index)

        if request_dest_index is None:
            req_d = torch.zeros(
                batch_size,
                num_requests,
                self.d_model,
                device=device,
                dtype=vehicle_features.dtype,
            )
        else:
            req_d = self._gather_nodes(node_emb, request_dest_index)

        # Broadcast request node embs across vehicles: (B, K, R, D)
        req_o_bk = req_o.unsqueeze(1).expand(-1, num_vehicles, -1, -1)
        req_d_bk = req_d.unsqueeze(1).expand(-1, num_vehicles, -1, -1)

        env_for_vehicles = environment_features.unsqueeze(1).expand(
            -1, num_vehicles, -1
        )
        vehicle_inputs = torch.cat(
            [vehicle_features, env_for_vehicles, veh_loc], dim=-1
        )
        vehicle_embeddings = self.vehicle_norm(self.vehicle_proj(vehicle_inputs))

        key_padding_mask = None
        if vehicle_padding_mask is not None:
            key_padding_mask = ~vehicle_padding_mask

        coordinated = vehicle_embeddings
        for layer in self.coordinator_layers:
            coordinated = layer(coordinated, key_padding_mask=key_padding_mask)

        request_inputs = torch.cat(
            [request_features, pairwise_features, req_o_bk, req_d_bk], dim=-1
        )
        request_embeddings = self.request_norm(self.request_feat_proj(request_inputs))

        if request_padding_mask is not None:
            # Zero padded request keys so mean-pool critic is not polluted.
            request_embeddings = request_embeddings * request_padding_mask.unsqueeze(
                1
            ).unsqueeze(-1).to(dtype=request_embeddings.dtype)

        queries = self.q_proj(coordinated)  # (B, K, D)
        keys = self.k_proj(request_embeddings)  # (B, K, R, D)
        attention_scores = torch.einsum("bkd,bkrd->bkr", queries, keys) / (
            self.d_model ** 0.5
        )

        # Pad request logit axis to max_requests so action indices stay stable.
        if num_requests < self.max_requests:
            pad = attention_scores.new_zeros(
                batch_size, num_vehicles, self.max_requests - num_requests
            )
            attention_scores = torch.cat([attention_scores, pad], dim=-1)

        specials = self.special_head(coordinated)  # (B, K, 2) → STAY, IDLE
        logits = torch.cat([attention_scores, specials], dim=-1)

        if request_padding_mask is not None:
            denom = request_padding_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            # Mean over valid requests only; embeddings already zeroed for pads.
            sum_req = request_embeddings.sum(dim=2)
            avg_request = sum_req / denom.unsqueeze(-1)
        else:
            avg_request = request_embeddings.mean(dim=2)

        critic_input = torch.cat([coordinated, avg_request, env_for_vehicles], dim=-1)
        values = self.critic_mlp(critic_input)

        if vehicle_padding_mask is not None:
            values = values * vehicle_padding_mask.unsqueeze(-1).to(dtype=values.dtype)

        return logits, values


def forward_with_mask(
    model: FleetRouterModel,
    vehicle_features: torch.Tensor,
    request_features: torch.Tensor,
    pairwise_features: torch.Tensor,
    environment_features: torch.Tensor,
    vehicle_padding_mask: torch.Tensor | None = None,
    request_padding_mask: torch.Tensor | None = None,
    **graph_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run model and apply legal-action masking. Returns masked logits, values, mask."""
    logits, values = model(
        vehicle_features,
        request_features,
        pairwise_features,
        environment_features,
        vehicle_padding_mask=vehicle_padding_mask,
        request_padding_mask=request_padding_mask,
        **graph_kwargs,
    )
    mask = build_action_mask(
        request_features,
        pairwise_features,
        request_padding_mask=request_padding_mask,
        vehicle_padding_mask=vehicle_padding_mask,
    )
    # Mask padded request slots even if availability flags are stale.
    if pairwise_features.dim() == 3:
        num_req = pairwise_features.size(1)
    else:
        num_req = pairwise_features.size(2)

    mask = mask.clone()
    if request_padding_mask is not None:
        req_pad = request_padding_mask
        if req_pad.dim() == 1:
            req_pad = req_pad.unsqueeze(0)
        slot_ok = req_pad.unsqueeze(1).expand(-1, mask.size(1), -1)
        mask[:, :, :num_req] = mask[:, :, :num_req] & slot_ok
    if num_req < MAX_REQUESTS:
        mask[:, :, num_req:MAX_REQUESTS] = False
    return apply_action_mask(logits, mask), values, mask


# Re-export action indices for trainers.
__all__ = [
    "FleetRouterModel",
    "GraphEncoder",
    "MeanGNNLayer",
    "VehicleTransformerBlock",
    "forward_with_mask",
    "scatter_mean",
    "ACTION_STAY",
    "ACTION_IDLE",
    "NUM_ACTIONS",
]
