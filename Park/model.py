from __future__ import annotations

import torch
import torch.nn as nn

from Park.training.features import (
    D_MODEL,
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    apply_action_mask,
    build_action_mask,
)


class ParkRouterModel(nn.Module):
    """Single-party pointer actor-critic (no guest-axis coordination)."""

    def __init__(
        self,
        guest_feat_dim: int,
        num_rides: int,
        ride_dynamic_feat_dim: int,
        environment_dynamic_feat_dim: int,
        d_model: int = D_MODEL,
        num_actions: int | None = None,
    ):
        super().__init__()

        self.num_rides = num_rides
        self.num_actions = num_actions or (num_rides + 2)
        self.d_model = d_model

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        guest: (B, guest_feat)
        ride:  (B, R, ride_feat)
        env:   (B, env_feat)

        Returns:
            logits: (B, num_actions)
            values: (B, 1)
        """
        batch_size = guest_dynamic_features.size(0)
        num_rides = ride_dynamic_features.size(1)

        guest_inputs = torch.cat(
            [guest_dynamic_features, environment_dynamic_features], dim=-1
        )
        guest_embeddings = self.guest_norm(self.guest_proj(guest_inputs))

        ride_ids = torch.arange(num_rides, device=ride_dynamic_features.device)
        ride_ids = ride_ids.view(1, num_rides).expand(batch_size, -1)
        ride_embeddings = self.ride_norm(
            self.ride_id_embed(ride_ids) + self.ride_feat_proj(ride_dynamic_features)
        )

        queries = self.q_proj(guest_embeddings)  # (B, D)
        keys = self.k_proj(ride_embeddings)  # (B, R, D)
        attention_scores = torch.einsum("bd,brd->br", queries, keys) / (self.d_model ** 0.5)
        exit_idle = self.exit_idle_head(guest_embeddings)
        logits = torch.cat([attention_scores, exit_idle], dim=-1)

        avg_ride = ride_embeddings.mean(dim=1)  # (B, D)
        critic_input = torch.cat(
            [guest_embeddings, avg_ride, environment_dynamic_features], dim=-1
        )
        values = self.critic_mlp(critic_input)  # (B, 1)
        return logits, values


def obs_flat_to_tensors(
    obs_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert flattened observation vectors to single-party model inputs.

    obs_flat: (B, FLAT_OBS_DIM) → guest (B, …), ride (B, R, …), env (B, …)
    """
    if obs_flat.dim() == 1:
        obs_flat = obs_flat.unsqueeze(0)
    guest_end = GUEST_FEAT_DIM
    ride_end = guest_end + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    guest = obs_flat[:, :guest_end]
    ride = obs_flat[:, guest_end:ride_end].view(-1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = obs_flat[:, ride_end:]
    return guest, ride, env


def forward_with_mask(
    model: ParkRouterModel,
    guest: torch.Tensor,
    ride: torch.Tensor,
    env: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run model and apply legal-action masking. Returns masked logits, values, mask."""
    logits, values = model(guest, ride, env)
    mask = build_action_mask(guest, ride, env)
    return apply_action_mask(logits, mask), values, mask
