from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from Park.training.features import (
    D_MODEL,
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    ROUTE_PAD,
    apply_action_mask,
    build_action_mask,
    build_tail_ride_mask,
    route_k as default_route_k,
)


@dataclass
class RouteOutput:
    """Autoregressive route decode result."""

    routes: torch.Tensor  # (B, K) int64; ROUTE_PAD after exit/idle
    log_prob: torch.Tensor  # (B,)
    entropy: torch.Tensor  # (B,) slot-weighted
    values: torch.Tensor  # (B, 1)
    slot0_logits: torch.Tensor  # (B, A) masked
    slot0_mask: torch.Tensor  # (B, A)
    slot_logits: torch.Tensor  # (B, K, A) masked; ride-only slots pad exit/idle as -inf
    slot_masks: torch.Tensor  # (B, K, A) bool; ride-only slots pad exit/idle as False


class ParkRouterModel(nn.Module):
    """Single-party pointer actor-critic with length-K autoregressive route decoder."""

    def __init__(
        self,
        guest_feat_dim: int,
        num_rides: int,
        ride_dynamic_feat_dim: int,
        environment_dynamic_feat_dim: int,
        d_model: int = D_MODEL,
        num_actions: int | None = None,
        route_k: int | None = None,
    ):
        super().__init__()

        self.num_rides = num_rides
        self.num_actions = num_actions or (num_rides + 2)
        self.d_model = d_model
        self.route_k = int(route_k) if route_k is not None else default_route_k()

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

        # Action embeds for decoder steps: rides + exit + idle
        self.action_embed = nn.Embedding(self.num_actions, d_model)
        self.decoder_rnn = nn.GRUCell(d_model, d_model)

        self.critic_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + environment_dynamic_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _encode(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        avg_ride = ride_embeddings.mean(dim=1)
        critic_input = torch.cat(
            [guest_embeddings, avg_ride, environment_dynamic_features], dim=-1
        )
        values = self.critic_mlp(critic_input)
        return guest_embeddings, ride_embeddings, values

    def _pointer_logits(self, hidden: torch.Tensor, ride_embeddings: torch.Tensor) -> torch.Tensor:
        queries = self.q_proj(hidden)
        keys = self.k_proj(ride_embeddings)
        return torch.einsum("bd,brd->br", queries, keys) / (self.d_model ** 0.5)

    def _entropy_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        try:
            import Park.config as config

            weights = getattr(
                config,
                "PPO_ROUTE_ENTROPY_WEIGHTS",
                (1.0, 0.75, 0.5, 0.25, 0.15, 0.1),
            )
        except Exception:
            weights = (1.0, 0.75, 0.5, 0.25, 0.15, 0.1)
        w = torch.tensor(weights[: self.route_k], device=device, dtype=dtype)
        if w.numel() < self.route_k:
            pad = torch.full(
                (self.route_k - w.numel(),),
                float(w[-1].item()) if w.numel() else 0.1,
                device=device,
                dtype=dtype,
            )
            w = torch.cat([w, pad], dim=0)
        return w

    def forward(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slot-0 logits + value (BC / simple callers). Does not run the full route decode."""
        guest_embeddings, ride_embeddings, values = self._encode(
            guest_dynamic_features,
            ride_dynamic_features,
            environment_dynamic_features,
        )
        attention_scores = self._pointer_logits(guest_embeddings, ride_embeddings)
        exit_idle = self.exit_idle_head(guest_embeddings)
        logits = torch.cat([attention_scores, exit_idle], dim=-1)
        return logits, values

    def forward_route(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
        routes: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
        force_first: torch.Tensor | None = None,
    ) -> RouteOutput:
        """Autoregressive route decode.

        routes: optional teacher-forced (B, K) with ROUTE_PAD after exit/idle.
        force_first: optional (B,) int64; values >= 0 pin slot 0 (if legal) then
            greedy/sample the remaining slots. Ignored when ``routes`` is set.
            Use -1 per batch row for no force. Companion ONNX always passes this.
        """
        guest_embeddings, ride_embeddings, values = self._encode(
            guest_dynamic_features,
            ride_dynamic_features,
            environment_dynamic_features,
        )
        batch = guest_embeddings.size(0)
        device = guest_embeddings.device
        dtype = guest_embeddings.dtype
        k = self.route_k

        slot0_mask = build_action_mask(
            guest_dynamic_features,
            ride_dynamic_features,
            environment_dynamic_features,
        )
        attention_scores = self._pointer_logits(guest_embeddings, ride_embeddings)
        exit_idle = self.exit_idle_head(guest_embeddings)
        slot0_logits = apply_action_mask(
            torch.cat([attention_scores, exit_idle], dim=-1), slot0_mask
        )

        ent_w = self._entropy_weights(device, dtype)
        hidden = guest_embeddings
        picked = torch.zeros(batch, self.num_rides, dtype=torch.bool, device=device)
        active = torch.ones(batch, dtype=torch.bool, device=device)

        route_list: list[torch.Tensor] = []
        slot_logits_list: list[torch.Tensor] = []
        slot_masks_list: list[torch.Tensor] = []
        log_prob = torch.zeros(batch, device=device, dtype=dtype)
        entropy = torch.zeros(batch, device=device, dtype=dtype)

        if force_first is None:
            force_first = torch.full((batch,), -1, dtype=torch.long, device=device)
        else:
            force_first = force_first.to(device=device, dtype=torch.long).reshape(batch)

        for step in range(k):
            if step == 0:
                logits = slot0_logits
                mask = slot0_mask
                pad_logits = logits
                pad_mask = mask
            else:
                ride_logits = self._pointer_logits(hidden, ride_embeddings)
                mask_r = build_tail_ride_mask(ride_dynamic_features, picked)
                # If nothing legal remains, allow any unfinished open ride (still masked by picked).
                none = ~mask_r.any(dim=-1)
                fallback = (ride_dynamic_features[..., 2] > 0.5) & (~picked)
                # Always blend (identity when any legal) so ONNX traces the path;
                # use &/| not bool torch.where — ORT has no Where(bool).
                none_b = none.unsqueeze(-1)
                mask_r = (fallback & none_b) | (mask_r & ~none_b)
                logits = apply_action_mask(ride_logits, mask_r)
                # Pad to full action dim unused; sample in ride space only.
                mask = mask_r
                pad_logits = torch.full(
                    (batch, self.num_actions),
                    -1.0e9,
                    device=device,
                    dtype=dtype,
                )
                pad_logits[:, : self.num_rides] = logits
                pad_mask = torch.zeros(
                    batch, self.num_actions, dtype=torch.bool, device=device
                )
                pad_mask[:, : self.num_rides] = mask

            slot_logits_list.append(pad_logits)
            slot_masks_list.append(pad_mask)

            dist = torch.distributions.Categorical(logits=logits)

            if routes is not None:
                raw = routes[:, step]
                # Padded slots: keep a legal dummy so log_prob is defined; ``active`` zeros it.
                if step == 0:
                    action = raw.clamp(0, self.num_actions - 1)
                else:
                    action = raw.clamp(0, self.num_rides - 1)
                illegal = ~mask.gather(1, action.unsqueeze(1)).squeeze(1)
                if illegal.any():
                    legal_idx = mask.float().argmax(dim=-1)
                    action = torch.where(illegal | (raw < 0), legal_idx, action)
            else:
                if deterministic:
                    natural = logits.argmax(dim=-1)
                else:
                    natural = dist.sample()
                if step == 0:
                    # Pin slot 0 when force_first >= 0 and legal; else natural.
                    # Always-on tensor ops so ONNX traces the force branch.
                    forced = force_first.clamp(0, self.num_actions - 1)
                    use_force = (force_first >= 0) & mask.gather(
                        1, forced.unsqueeze(1)
                    ).squeeze(1)
                    action = torch.where(use_force, forced, natural)
                else:
                    action = natural

            step_logp = dist.log_prob(action)
            step_ent = dist.entropy()
            log_prob = log_prob + torch.where(active, step_logp, torch.zeros_like(step_logp))
            entropy = entropy + torch.where(
                active, ent_w[step] * step_ent, torch.zeros_like(step_ent)
            )

            if step == 0:
                stored = action
            else:
                stored = torch.where(active, action, torch.full_like(action, ROUTE_PAD))
            route_list.append(stored)

            # Update active / picked / hidden for next slot.
            # Always apply tensor updates (no `if is_ride.any()`): torch.onnx
            # tracing freezes Python control flow from the example inputs, so a
            # soft-close export path would omit picked/GRU updates and repeat
            # the same ride for every slot at inference.
            is_ride = (action < self.num_rides) & active
            ride_actions = action.clamp(0, self.num_rides - 1)
            picked = picked | (
                torch.nn.functional.one_hot(ride_actions, self.num_rides).bool()
                & is_ride.unsqueeze(-1)
            )
            emb = self.action_embed(action.clamp(0, self.num_actions - 1))
            new_hidden = self.decoder_rnn(emb, hidden)
            hidden = torch.where(is_ride.unsqueeze(-1), new_hidden, hidden)

            # Exit/idle (or inactive) → stop; only ride commits continue.
            active = is_ride

        routes_out = torch.stack(route_list, dim=1)
        if routes is not None:
            # Preserve PAD from teacher labels for non-active slots
            pad_mask = routes < 0
            routes_out = torch.where(pad_mask, routes, routes_out)

        return RouteOutput(
            routes=routes_out,
            log_prob=log_prob,
            entropy=entropy,
            values=values,
            slot0_logits=slot0_logits,
            slot0_mask=slot0_mask,
            slot_logits=torch.stack(slot_logits_list, dim=1),
            slot_masks=torch.stack(slot_masks_list, dim=1),
        )


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
    """Run model slot-0 head and apply legal-action masking."""
    logits, values = model(guest, ride, env)
    mask = build_action_mask(guest, ride, env)
    return apply_action_mask(logits, mask), values, mask


def forward_route_with_mask(
    model: ParkRouterModel,
    guest: torch.Tensor,
    ride: torch.Tensor,
    env: torch.Tensor,
    routes: torch.Tensor | None = None,
    *,
    deterministic: bool = False,
    force_first: torch.Tensor | None = None,
) -> RouteOutput:
    """Full route decode (masking applied inside the model)."""
    return model.forward_route(
        guest,
        ride,
        env,
        routes=routes,
        deterministic=deterministic,
        force_first=force_first,
    )
