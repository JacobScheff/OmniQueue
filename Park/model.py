from __future__ import annotations

import torch
import torch.nn as nn

from training.features import (
    D_MODEL,
    ENV_DYNAMIC_FEAT_DIM,
    GUEST_FEAT_DIM,
    NUM_ATTN_HEADS,
    NUM_RIDES,
    NUM_TRANSFORMER_LAYERS,
    RIDE_DYNAMIC_FEAT_DIM,
    apply_action_mask,
    build_action_mask,
)


class GuestTransformerBlock(nn.Module):
    """Self-attention + FFN over the guest/party axis (coordinator stack)."""

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
        # key_padding_mask: (B, G) True = PAD (ignored), matching nn.MultiheadAttention
        attn_out, _ = self.attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class ParkRouterModel(nn.Module):
    def __init__(
        self,
        guest_feat_dim: int,
        num_rides: int,
        ride_dynamic_feat_dim: int,
        environment_dynamic_feat_dim: int,
        d_model: int = D_MODEL,
        num_actions: int | None = None,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        num_heads: int = NUM_ATTN_HEADS,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.num_rides = num_rides
        self.num_actions = num_actions or (num_rides + 2)
        self.d_model = d_model
        self.num_layers = num_layers

        self.ride_id_embed = nn.Embedding(num_rides, d_model)
        self.ride_feat_proj = nn.Sequential(
            nn.Linear(ride_dynamic_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.ride_norm = nn.LayerNorm(d_model)

        self.guest_proj = nn.Sequential(
            nn.Linear(guest_feat_dim + environment_dynamic_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.guest_norm = nn.LayerNorm(d_model)

        self.coordinator_layers = nn.ModuleList(
            [GuestTransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.exit_idle_head = nn.Linear(d_model, 2)

        self.critic_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + environment_dynamic_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
        guest_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        guest: (B, G, guest_feat)
        ride:  (B, G, R, ride_feat) or (B, R, ride_feat) for G=1 shared rides
        env:   (B, env_feat)
        guest_padding_mask: (B, G) True = valid guest (False = pad)

        Returns:
            logits: (B, G, num_actions)
            values: (B, G, 1) per-party critic estimates
        """
        if ride_dynamic_features.dim() == 3:
            # (B, R, F) → broadcast across a single guest axis
            ride_dynamic_features = ride_dynamic_features.unsqueeze(1)

        batch_size, num_guests, _ = guest_dynamic_features.size()
        num_rides = ride_dynamic_features.size(2)

        env_for_guests = environment_dynamic_features.unsqueeze(1).expand(-1, num_guests, -1)
        guest_inputs = torch.cat([guest_dynamic_features, env_for_guests], dim=-1)
        guest_embeddings = self.guest_norm(self.guest_proj(guest_inputs))

        # MultiheadAttention pad mask: True = ignore
        key_padding_mask = None
        if guest_padding_mask is not None:
            key_padding_mask = ~guest_padding_mask

        coordinated = guest_embeddings
        for layer in self.coordinator_layers:
            coordinated = layer(coordinated, key_padding_mask=key_padding_mask)

        ride_ids = torch.arange(num_rides, device=ride_dynamic_features.device)
        ride_ids = ride_ids.view(1, 1, num_rides).expand(batch_size, num_guests, -1)
        ride_embeddings = self.ride_norm(
            self.ride_id_embed(ride_ids) + self.ride_feat_proj(ride_dynamic_features)
        )

        queries = self.q_proj(coordinated)  # (B, G, D)
        keys = self.k_proj(ride_embeddings)  # (B, G, R, D)
        # Per-guest pointer scores: (B, G, R)
        attention_scores = torch.einsum("bgd,bgrd->bgr", queries, keys) / (self.d_model ** 0.5)
        exit_idle = self.exit_idle_head(coordinated)
        logits = torch.cat([attention_scores, exit_idle], dim=-1)

        avg_ride = ride_embeddings.mean(dim=2)  # (B, G, D)
        critic_input = torch.cat([coordinated, avg_ride, env_for_guests], dim=-1)
        values = self.critic_mlp(critic_input)  # (B, G, 1)

        if guest_padding_mask is not None:
            values = values * guest_padding_mask.unsqueeze(-1).to(dtype=values.dtype)

        return logits, values


def obs_flat_to_tensors(
    obs_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert flattened observation vectors to model inputs (batch of G=1 parties).

    obs_flat: (B, FLAT_OBS_DIM) → guest (B, 1, …), ride (B, 1, R, …), env (B, …)
    """
    guest_end = GUEST_FEAT_DIM
    ride_end = guest_end + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    guest = obs_flat[:, :guest_end].unsqueeze(1)
    ride = obs_flat[:, guest_end:ride_end].view(-1, 1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = obs_flat[:, ride_end:]
    return guest, ride, env


def obs_group_to_tensors(
    obs_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a co-timed party group to a single batch item (B=1, G=N).

    obs_flat: (G, FLAT_OBS_DIM)
    """
    if obs_flat.dim() == 1:
        obs_flat = obs_flat.unsqueeze(0)
    guest_end = GUEST_FEAT_DIM
    ride_end = guest_end + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    guest = obs_flat[:, :guest_end].unsqueeze(0)  # (1, G, F)
    ride = obs_flat[:, guest_end:ride_end].view(1, -1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = obs_flat[:1, ride_end:]  # shared park time from first party
    return guest, ride, env


def forward_with_mask(
    model: ParkRouterModel,
    guest: torch.Tensor,
    ride: torch.Tensor,
    env: torch.Tensor,
    guest_padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run model and apply legal-action masking. Returns masked logits, values, mask."""
    logits, values = model(guest, ride, env, guest_padding_mask=guest_padding_mask)
    mask = build_action_mask(guest, ride, env)
    if guest_padding_mask is not None:
        mask = mask & guest_padding_mask.unsqueeze(-1)
    return apply_action_mask(logits, mask), values, mask
