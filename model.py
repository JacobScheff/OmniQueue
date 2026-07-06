from __future__ import annotations

import torch
import torch.nn as nn

from training.features import (
    ENV_DYNAMIC_FEAT_DIM,
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
)


class ParkRouterModel(nn.Module):
    def __init__(
        self,
        guest_feat_dim: int,
        num_rides: int,
        ride_dynamic_feat_dim: int,
        environment_dynamic_feat_dim: int,
        d_model: int = 128,
        num_actions: int | None = None,
    ):
        super().__init__()

        self.num_rides = num_rides
        self.num_actions = num_actions or (num_rides + 2)
        self.d_model = d_model

        self.ride_embed = nn.Embedding(
            num_rides,
            d_model - ride_dynamic_feat_dim - environment_dynamic_feat_dim,
        )
        self.guest_embed = nn.Linear(guest_feat_dim + environment_dynamic_feat_dim, d_model)

        self.coordinator = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.exit_idle_head = nn.Linear(d_model, 2)

        self.critic_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + environment_dynamic_feat_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_guests, _ = guest_dynamic_features.size()

        env_for_rides = environment_dynamic_features.unsqueeze(1).expand(-1, self.num_rides, -1)
        env_for_guests = environment_dynamic_features.unsqueeze(1).expand(-1, num_guests, -1)

        ride_ids = torch.arange(self.num_rides, device=ride_dynamic_features.device).expand(batch_size, -1)
        ride_learned_embeddings = self.ride_embed(ride_ids)

        ride_embeddings = torch.cat([ride_learned_embeddings, ride_dynamic_features, env_for_rides], dim=-1)

        guest_inputs = torch.cat([guest_dynamic_features, env_for_guests], dim=-1)
        guest_embeddings = self.guest_embed(guest_inputs)

        coordinate_attn, _ = self.coordinator(guest_embeddings, guest_embeddings, guest_embeddings)
        coordinated_guests = guest_embeddings + coordinate_attn

        queries = self.q_proj(coordinated_guests)
        keys = self.k_proj(ride_embeddings)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_model ** 0.5)
        exit_idle = self.exit_idle_head(coordinated_guests)
        logits = torch.cat([attention_scores, exit_idle], dim=-1)

        avg_guest_embedding = coordinated_guests.mean(dim=1)
        avg_ride_embedding = ride_embeddings.mean(dim=1)
        critic_input = torch.cat([avg_guest_embedding, avg_ride_embedding, environment_dynamic_features], dim=-1)

        global_value = self.critic_mlp(critic_input)

        return logits, global_value


def obs_flat_to_tensors(
    obs_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert flattened observation vectors to model inputs (batch, 1 guest)."""
    guest_end = GUEST_FEAT_DIM
    ride_end = guest_end + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    guest = obs_flat[:, :guest_end].unsqueeze(1)
    ride = obs_flat[:, guest_end:ride_end].view(-1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = obs_flat[:, ride_end:]
    return guest, ride, env
