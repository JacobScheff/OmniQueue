import torch
import torch.nn as nn


class ParkRouterModel(nn.Module):
    def __init__(
        self,
        guest_feat_dim,
        num_rides,
        ride_dynamic_feat_dim,
        environment_dynamic_feat_dim,
        d_model=128,
        num_actions=None,
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

    def forward(self, guest_dynamic_features, ride_dynamic_features, environment_dynamic_features):
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

    def act(self, guest_dynamic_features, ride_dynamic_features, environment_dynamic_features):
        logits, value = self.forward(guest_dynamic_features, ride_dynamic_features, environment_dynamic_features)
        dist = torch.distributions.Categorical(logits=logits[:, 0, :])
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value.squeeze(-1), dist.entropy()


def obs_flat_to_tensors(obs_flat: torch.Tensor):
    """Convert flattened observation vectors to model inputs (batch, 1 guest)."""
    guest_dim = 45
    ride_dim = 35 * 5
    guest = obs_flat[:, :guest_dim].unsqueeze(1)
    ride = obs_flat[:, guest_dim : guest_dim + ride_dim].view(-1, 35, 5)
    env = obs_flat[:, guest_dim + ride_dim :]
    return guest, ride, env


if __name__ == "__main__":
    model = ParkRouterModel(guest_feat_dim=45, num_rides=35, ride_dynamic_feat_dim=5, environment_dynamic_feat_dim=4)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
