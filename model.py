import torch
import torch.nn as nn
import torch.nn.functional as F

class ParkRouterModel(nn.Module):
    def __init__(self, guest_feat_dim, num_rides, ride_dynamic_feat_dim, environment_dynamic_feat_dim, d_model=128):
        super().__init__()

        self.num_rides = num_rides
        self.d_model = d_model

        # TODO: Add environment features to the model. Currently, they are not used.

        # Guests each have their own individual features and do not persist, so no pre-learned information from embeddings
        # Rides are permanent and thus have their own learned information (such as breakdown rates, positions/distances to other rides, etc.) along with dynamic features (such as current wait times, number of guests heading to the ride, etc.)
        # Environment features are global and are dynamic (time of day, weather, etc.) and are shared across all rides and guests

        self.ride_embed = nn.Embedding(num_rides, d_model - ride_dynamic_feat_dim - environment_dynamic_feat_dim) # Embedding features(d_model - dynamic features) + dynamic features = d_model
        self.guest_embed = nn.Linear(guest_feat_dim + environment_dynamic_feat_dim, d_model) # Map guest dynamic features to d_model

        # Guest-to-Guest Coordinator
        self.coordinator = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        # Cross-attention between guests and rides
        self.q_proj = nn.Linear(d_model, d_model) # Queries from Guests
        self.k_proj = nn.Linear(d_model, d_model) # Keys from Rides

        # Critic Head (Global Value of the Park)
        self.critic_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, guest_dynamic_features, ride_dynamic_features, environment_dynamic_features):
        # --- Actor --- #

        # Embed the rides
        ride_learned_embeddings = self.ride_embed(torch.arange(ride_dynamic_features.size(0), device=ride_dynamic_features.device))  # (num_rides, d_model - ride_dynamic_feat_dim)

        # Concatenate learned embeddings with dynamic features
        ride_embeddings = torch.cat([ride_learned_embeddings, ride_dynamic_features], dim=-1) # (num_rides, d_model - ride_dynamic_feat_dim) + (num_rides, ride_dynamic_feat_dim) = (num_rides, d_model)

        # Expand ride embeddings to match the batch size of guests
        ride_embeddings = ride_embeddings.unsqueeze(0).expand(guest_dynamic_features.size(0), -1, -1)  # (batch_size, num_rides, d_model)

        # Embed the guests' dynamic features
        guest_embeddings = self.guest_embed(torch.cat([guest_dynamic_features, environment_dynamic_features], dim=-1))  # (batch_size, num_guests, d_model)

        # Guest-to-Guest Coordination (V is unused, since only the attentinon is what matters)
        coordinated_guests, _ = self.coordinator(guest_embeddings, guest_embeddings, guest_embeddings)  # (batch_size, num_guests, d_model)
        
        # Cross-attention between guests and rides
        queries = self.q_proj(coordinated_guests)  # (batch_size, num_guests, d_model)
        keys = self.k_proj(ride_embeddings)  # (num_rides, d_model)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_model ** 0.5)  # (batch_size, num_guests, num_rides)

        # --- Critic --- #
        # Global value of the park based on the average of the coordinated guest embeddings and the average of the ride embeddings
        avg_guest_embedding = coordinated_guests.mean(dim=1)  # (batch_size, d_model)
        avg_ride_embedding = ride_embeddings.mean(dim=1)  # (batch_size, d_model)
        critic_input = torch.cat([avg_guest_embedding, avg_ride_embedding], dim=-1)  # (batch_size, d_model * 2)
        
        global_value = self.critic_mlp(critic_input)  # (batch_size, 1)


        return attention_scores, global_value