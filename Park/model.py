from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from Park.training.features import (
    CANDIDATE_M,
    D_MODEL,
    GUEST_FEAT_DIM,
    N_CROSS_HEADS,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    RIDE_FEAT_ETA,
    RIDE_FEAT_MUST_DO,
    RIDE_FEAT_OPEN,
    RIDE_FEAT_UNFINISHED_PREF,
    RIDE_FEAT_WAIT,
    RIDE_FEAT_WAIT_VS_MEAN,
    RIDE_FEAT_WALK,
    ROUTE_PAD,
    apply_action_mask,
    build_action_mask,
    build_tail_ride_mask,
    candidate_m as default_candidate_m,
    route_k as default_route_k,
)


def _ride_walk_norm_matrix() -> torch.Tensor:
    """(R, R) walk features in obs units (sec/3600, capped), at BASE_WALKING_SPEED."""
    try:
        from Park.training.route_reward import ride_walk_matrix_sec

        mat = ride_walk_matrix_sec()
        t = torch.as_tensor(mat, dtype=torch.float32)
        return t.clamp(min=0.0, max=3600.0) / 3600.0
    except Exception:
        return torch.zeros(NUM_RIDES, NUM_RIDES, dtype=torch.float32)


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


@dataclass
class RouteOutput:
    """Rank-then-route decode result."""

    routes: torch.Tensor  # (B, K) int64; ROUTE_PAD after exit/idle
    log_prob: torch.Tensor  # (B,) Stage A + decoder
    entropy: torch.Tensor  # (B,) Stage A + light decoder
    values: torch.Tensor  # (B, 1)
    slot0_logits: torch.Tensor  # (B, A) Stage A masked
    slot0_mask: torch.Tensor  # (B, A)
    slot_logits: torch.Tensor  # (B, K, A) masked
    slot_masks: torch.Tensor  # (B, K, A) bool
    stage_a_logits: torch.Tensor | None = None  # alias of slot0 when set
    stage_a_mask: torch.Tensor | None = None


class GuestRideCrossAttention(nn.Module):
    """Guest queries rides (multi-head); rides get a guest-broadcast residual."""

    def __init__(self, d_model: int, n_heads: int = N_CROSS_HEADS):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_g = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)
        self.ride_guest_proj = nn.Linear(d_model, d_model)

    def forward(
        self, guest_emb: torch.Tensor, ride_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = guest_emb.unsqueeze(1)
        attn_out, _ = self.mha(g, ride_emb, ride_emb, need_weights=False)
        guest_ctx = self.norm_g(guest_emb + attn_out.squeeze(1))
        ride_ctx = self.norm_r(
            ride_emb + self.ride_guest_proj(guest_emb).unsqueeze(1)
        )
        return guest_ctx, ride_ctx


class RankRouteModel(nn.Module):
    """Two-stage personal planner: Stage A ride scorer + Stage B candidate route decoder."""

    def __init__(
        self,
        guest_feat_dim: int,
        num_rides: int,
        ride_dynamic_feat_dim: int,
        environment_dynamic_feat_dim: int,
        d_model: int = D_MODEL,
        num_actions: int | None = None,
        route_k: int | None = None,
        candidate_m: int | None = None,
        n_cross_heads: int = N_CROSS_HEADS,
    ):
        super().__init__()

        self.num_rides = num_rides
        self.num_actions = num_actions or (num_rides + 2)
        self.d_model = d_model
        self.route_k = int(route_k) if route_k is not None else default_route_k()
        self.candidate_m = (
            int(candidate_m) if candidate_m is not None else default_candidate_m()
        )
        self.env_dim = environment_dynamic_feat_dim

        self.ride_id_embed = nn.Embedding(num_rides, d_model)
        self.ride_feat_proj = _mlp(
            [ride_dynamic_feat_dim, d_model * 2, d_model * 2, d_model]
        )
        self.ride_norm = nn.LayerNorm(d_model)

        self.guest_proj = _mlp(
            [guest_feat_dim + environment_dynamic_feat_dim, d_model * 2, d_model * 2, d_model]
        )
        self.guest_norm = nn.LayerNorm(d_model)

        self.cross_attn = GuestRideCrossAttention(d_model, n_cross_heads)

        # Stage A: per-ride MLP on [ride_ctx; guest_ctx; wait; walk; pref; must_do; eta]
        score_in = d_model * 2 + 5
        self.ride_scorer = _mlp([score_in, d_model * 2, d_model * 2, d_model, 1])
        self.exit_idle_head = _mlp([d_model, d_model, 2])

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.score_bias = nn.Linear(1, 1)

        self.action_embed = nn.Embedding(self.num_actions, d_model)
        self.decoder_rnn = nn.GRUCell(d_model, d_model)
        self.cand_pool = nn.Linear(d_model, d_model)

        self.critic_mlp = _mlp(
            [
                d_model * 2 + environment_dynamic_feat_dim,
                d_model * 2,
                d_model * 2,
                d_model,
                1,
            ]
        )

        walk = _ride_walk_norm_matrix()
        if walk.shape != (num_rides, num_rides):
            walk = torch.zeros(num_rides, num_rides, dtype=torch.float32)
        self.register_buffer("ride_walk_norm", walk, persistent=True)

    def _encode_rides(self, ride_dynamic_features: torch.Tensor) -> torch.Tensor:
        batch_size = ride_dynamic_features.size(0)
        num_rides = ride_dynamic_features.size(1)
        ride_ids = torch.arange(num_rides, device=ride_dynamic_features.device)
        ride_ids = ride_ids.view(1, num_rides).expand(batch_size, -1)
        return self.ride_norm(
            self.ride_id_embed(ride_ids) + self.ride_feat_proj(ride_dynamic_features)
        )

    def _encode_context(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        guest_inputs = torch.cat(
            [guest_dynamic_features, environment_dynamic_features], dim=-1
        )
        guest_embeddings = self.guest_norm(self.guest_proj(guest_inputs))
        ride_embeddings = self._encode_rides(ride_dynamic_features)
        guest_ctx, ride_ctx = self.cross_attn(guest_embeddings, ride_embeddings)

        avg_ride = ride_ctx.mean(dim=1)
        critic_input = torch.cat(
            [guest_ctx, avg_ride, environment_dynamic_features], dim=-1
        )
        values = self.critic_mlp(critic_input)
        return guest_ctx, ride_ctx, values

    def _stage_a_logits(
        self,
        guest_ctx: torch.Tensor,
        ride_ctx: torch.Tensor,
        ride_feats: torch.Tensor,
        guest_feats: torch.Tensor,
        env_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_rides, _ = ride_ctx.shape
        wait = ride_feats[..., RIDE_FEAT_WAIT]
        walk = ride_feats[..., RIDE_FEAT_WALK]
        pref = ride_feats[..., RIDE_FEAT_UNFINISHED_PREF]
        must = ride_feats[..., RIDE_FEAT_MUST_DO]
        eta = ride_feats[..., RIDE_FEAT_ETA]
        scalars = torch.stack([wait, walk, pref, must, eta], dim=-1)
        guest_exp = guest_ctx.unsqueeze(1).expand(-1, num_rides, -1)
        scorer_in = torch.cat([ride_ctx, guest_exp, scalars], dim=-1)
        ride_scores = self.ride_scorer(scorer_in).squeeze(-1)
        exit_idle = self.exit_idle_head(guest_ctx)
        logits = torch.cat([ride_scores, exit_idle], dim=-1)
        mask = build_action_mask(guest_feats, ride_feats, env_feats)
        return apply_action_mask(logits, mask), mask

    def _pointer_logits(
        self, hidden: torch.Tensor, cand_emb: torch.Tensor, cand_scores: torch.Tensor
    ) -> torch.Tensor:
        queries = self.q_proj(hidden)
        keys = self.k_proj(cand_emb)
        logits = torch.einsum("bd,bmd->bm", queries, keys) / (self.d_model**0.5)
        bias = self.score_bias(cand_scores.unsqueeze(-1)).squeeze(-1)
        return logits + bias

    def _entropy_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        try:
            import Park.config as config

            weights = getattr(
                config,
                "PPO_ROUTE_ENTROPY_WEIGHTS",
                (1.0, 0.75, 0.5, 0.25, 0.15),
            )
        except Exception:
            weights = (1.0, 0.75, 0.5, 0.25, 0.15)
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

    def _select_candidates(
        self,
        stage_a_logits: torch.Tensor,
        ride_feats: torch.Tensor,
        slot0_mask: torch.Tensor,
        forced_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cand_idx (B,M), cand_emb placeholder indices, legal cand mask.

        Must-dos that are legal are forced into the set; remaining filled by top π_A.
        """
        batch = stage_a_logits.size(0)
        m = min(self.candidate_m, self.num_rides)
        device = stage_a_logits.device
        ride_logits = stage_a_logits[:, : self.num_rides]
        ride_mask = slot0_mask[:, : self.num_rides]
        # Large negative for illegal so they never win top-k
        scores = ride_logits.masked_fill(~ride_mask, -1.0e9)
        must = (ride_feats[..., RIDE_FEAT_MUST_DO] > 0.5) & ride_mask
        # Boost must-dos into the top-M
        boosted = scores + must.to(scores.dtype) * 1.0e6
        if forced_first is not None:
            ff = forced_first.clamp(0, self.num_rides - 1)
            is_ride = forced_first < self.num_rides
            boost_row = torch.zeros_like(boosted)
            boost_row.scatter_(1, ff.unsqueeze(1), 1.0e6)
            boosted = boosted + boost_row * is_ride.unsqueeze(-1).to(boosted.dtype)

        _top_scores, cand_idx = torch.topk(boosted, k=m, dim=-1)
        # Cand legal if original ride was legal under slot0 mask
        cand_legal = ride_mask.gather(1, cand_idx)
        # Ensure at least one legal: fall back to argmax of scores
        none = ~cand_legal.any(dim=-1)
        if none.any():
            fallback = scores.argmax(dim=-1)
            cand_idx = cand_idx.clone()
            cand_idx[none, 0] = fallback[none]
            cand_legal = ride_mask.gather(1, cand_idx)
        cand_scores = ride_logits.gather(1, cand_idx)
        return cand_idx, cand_scores, cand_legal

    def forward(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage A logits + value (BC / simple callers)."""
        guest_ctx, ride_ctx, values = self._encode_context(
            guest_dynamic_features,
            ride_dynamic_features,
            environment_dynamic_features,
        )
        logits, _mask = self._stage_a_logits(
            guest_ctx,
            ride_ctx,
            ride_dynamic_features,
            guest_dynamic_features,
            environment_dynamic_features,
        )
        return logits, values

    def score_stage_a(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masked Stage A logits, mask, and values (for CF / pref-rank)."""
        guest_ctx, ride_ctx, values = self._encode_context(
            guest_dynamic_features,
            ride_dynamic_features,
            environment_dynamic_features,
        )
        logits, mask = self._stage_a_logits(
            guest_ctx,
            ride_ctx,
            ride_dynamic_features,
            guest_dynamic_features,
            environment_dynamic_features,
        )
        return logits, mask, values

    def forward_route(
        self,
        guest_dynamic_features: torch.Tensor,
        ride_dynamic_features: torch.Tensor,
        environment_dynamic_features: torch.Tensor,
        routes: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
        force_first: torch.Tensor | None = None,
        temperature: float = 1.0,
        close_margin: float = 0.0,
        top_p: float = 1.0,
    ) -> RouteOutput:
        """Stage A commit + Stage B candidate route decode."""
        guest_ctx, ride_ctx, values = self._encode_context(
            guest_dynamic_features,
            ride_dynamic_features,
            environment_dynamic_features,
        )
        batch = guest_ctx.size(0)
        device = guest_ctx.device
        dtype = guest_ctx.dtype
        k = self.route_k

        ride_feats = ride_dynamic_features.to(dtype=dtype).clone()
        walk_table = self.ride_walk_norm.to(device=device, dtype=dtype)
        mean_wait = environment_dynamic_features[:, 1:2]

        stage_a_logits, slot0_mask = self._stage_a_logits(
            guest_ctx,
            ride_ctx,
            ride_feats,
            guest_dynamic_features,
            environment_dynamic_features,
        )
        slot0_logits = stage_a_logits

        ent_w = self._entropy_weights(device, dtype)
        log_prob = torch.zeros(batch, device=device, dtype=dtype)
        entropy = torch.zeros(batch, device=device, dtype=dtype)

        if force_first is None:
            force_first = torch.full((batch,), -1, dtype=torch.long, device=device)
        else:
            force_first = force_first.to(device=device, dtype=torch.long).reshape(batch)

        # ---- Slot 0 from Stage A ----
        a_logits = slot0_logits
        if temperature != 1.0 and temperature > 0:
            a_logits = a_logits / temperature
        dist_a = torch.distributions.Categorical(logits=a_logits)

        if routes is not None:
            action0 = routes[:, 0].clamp(0, self.num_actions - 1)
            illegal = ~slot0_mask.gather(1, action0.unsqueeze(1)).squeeze(1)
            if illegal.any():
                legal_idx = slot0_mask.float().argmax(dim=-1)
                action0 = torch.where(illegal | (routes[:, 0] < 0), legal_idx, action0)
        else:
            natural = self._sample_or_argmax(
                a_logits,
                dist_a,
                deterministic=deterministic,
                close_margin=close_margin,
                top_p=top_p,
            )
            forced = force_first.clamp(0, self.num_actions - 1)
            use_force = (force_first >= 0) & slot0_mask.gather(
                1, forced.unsqueeze(1)
            ).squeeze(1)
            action0 = torch.where(use_force, forced, natural)

        log_prob = log_prob + dist_a.log_prob(action0)
        entropy = entropy + dist_a.entropy()  # Stage A entropy (full weight)

        route_list: list[torch.Tensor] = [action0]
        slot_logits_list: list[torch.Tensor] = [slot0_logits]
        slot_masks_list: list[torch.Tensor] = [slot0_mask]

        is_ride0 = action0 < self.num_rides
        active = is_ride0.clone()

        # Build candidates conditioned on Stage A (+ forced commit)
        cand_idx, cand_scores, cand_legal = self._select_candidates(
            stage_a_logits,
            ride_feats,
            slot0_mask,
            torch.where(is_ride0, action0, torch.full_like(action0, -1)),
        )
        cand_emb = ride_ctx.gather(
            1, cand_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        # Init decoder hidden from guest + pooled candidates
        weights = F.softmax(
            cand_scores.masked_fill(~cand_legal, -1.0e9), dim=-1
        ).unsqueeze(-1)
        pooled = (cand_emb * weights).sum(dim=1)
        hidden = guest_ctx + self.cand_pool(pooled)

        picked = torch.zeros(batch, self.num_rides, dtype=torch.bool, device=device)
        ride_actions0 = action0.clamp(0, self.num_rides - 1)
        picked = picked | (
            F.one_hot(ride_actions0, self.num_rides).bool() & is_ride0.unsqueeze(-1)
        )
        emb0 = self.action_embed(action0.clamp(0, self.num_actions - 1))
        new_hidden = self.decoder_rnn(emb0, hidden)
        hidden = torch.where(is_ride0.unsqueeze(-1), new_hidden, hidden)

        # Refresh walk/ETA as if at commit ride
        walk_row = walk_table[ride_actions0].clone()
        zero_here = torch.zeros(batch, 1, device=device, dtype=dtype)
        walk_row = walk_row.scatter(1, ride_actions0.unsqueeze(1), zero_here)
        old_walk = ride_feats[..., RIDE_FEAT_WALK]
        new_walk = torch.where(is_ride0.unsqueeze(-1), walk_row, old_walk)
        ride_feats = ride_feats.clone()
        ride_feats[..., RIDE_FEAT_WALK] = new_walk
        wait = ride_feats[..., RIDE_FEAT_WAIT]
        ride_feats[..., RIDE_FEAT_ETA] = (new_walk + wait).clamp(0.0, 2.0)
        ride_feats[..., RIDE_FEAT_WAIT_VS_MEAN] = (wait - mean_wait).clamp(-1.0, 1.0)
        ride_ctx = self._encode_rides(ride_feats)
        # Re-apply guest broadcast lightly via residual from guest_ctx
        ride_ctx = ride_ctx + self.cross_attn.ride_guest_proj(guest_ctx).unsqueeze(1)
        cand_emb = ride_ctx.gather(
            1, cand_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        # ---- Tail slots from Stage B over candidates ----
        for step in range(1, k):
            # Map candidate logits → full ride logits
            cand_logits = self._pointer_logits(hidden, cand_emb, cand_scores)
            # Legal: in candidate set, open, unfinished, not picked
            tail_ok = build_tail_ride_mask(ride_feats, picked)  # (B, R)
            # Candidate position legal if ride legal
            cand_ride_legal = tail_ok.gather(1, cand_idx) & cand_legal
            none = ~cand_ride_legal.any(dim=-1)
            # Fallback: any open unpicked ride among candidates
            open_unpicked = (ride_feats[..., RIDE_FEAT_OPEN] > 0.5) & (~picked)
            fb = open_unpicked.gather(1, cand_idx)
            none_b = none.unsqueeze(-1)
            cand_ride_legal = (fb & none_b) | (cand_ride_legal & ~none_b)

            cand_logits_m = apply_action_mask(cand_logits, cand_ride_legal)
            pad_logits = torch.full(
                (batch, self.num_actions), -1.0e9, device=device, dtype=dtype
            )
            # Scatter candidate logits into ride positions
            pad_logits.scatter_(1, cand_idx, cand_logits_m)
            pad_mask = torch.zeros(
                batch, self.num_actions, dtype=torch.bool, device=device
            )
            pad_mask.scatter_(1, cand_idx, cand_ride_legal)

            slot_logits_list.append(pad_logits)
            slot_masks_list.append(pad_mask)

            dist = torch.distributions.Categorical(logits=cand_logits_m)

            if routes is not None:
                raw = routes[:, step]
                # Map teacher ride id → candidate slot (or best legal)
                # Find which cand matches raw
                match = cand_idx == raw.clamp(0, self.num_rides - 1).unsqueeze(1)
                has = match.any(dim=-1)
                cand_pos = match.float().argmax(dim=-1)
                legal_pos = cand_ride_legal.float().argmax(dim=-1)
                cand_pos = torch.where(has & (raw >= 0), cand_pos, legal_pos)
                action_pos = cand_pos
            else:
                if deterministic:
                    action_pos = cand_logits_m.argmax(dim=-1)
                else:
                    action_pos = dist.sample()

            action = cand_idx.gather(1, action_pos.unsqueeze(1)).squeeze(1)
            step_logp = dist.log_prob(action_pos)
            step_ent = dist.entropy()
            log_prob = log_prob + torch.where(
                active, step_logp, torch.zeros_like(step_logp)
            )
            entropy = entropy + torch.where(
                active,
                ent_w[step] * step_ent,
                torch.zeros_like(step_ent),
            )

            stored = torch.where(active, action, torch.full_like(action, ROUTE_PAD))
            route_list.append(stored)

            is_ride = active & (action < self.num_rides)
            ride_actions = action.clamp(0, self.num_rides - 1)
            picked = picked | (
                F.one_hot(ride_actions, self.num_rides).bool() & is_ride.unsqueeze(-1)
            )
            emb = self.action_embed(action.clamp(0, self.num_actions - 1))
            new_hidden = self.decoder_rnn(emb, hidden)
            hidden = torch.where(is_ride.unsqueeze(-1), new_hidden, hidden)

            walk_row = walk_table[ride_actions].clone()
            walk_row = walk_row.scatter(1, ride_actions.unsqueeze(1), zero_here)
            old_walk = ride_feats[..., RIDE_FEAT_WALK]
            new_walk = torch.where(is_ride.unsqueeze(-1), walk_row, old_walk)
            ride_feats = ride_feats.clone()
            ride_feats[..., RIDE_FEAT_WALK] = new_walk
            wait = ride_feats[..., RIDE_FEAT_WAIT]
            ride_feats[..., RIDE_FEAT_ETA] = (new_walk + wait).clamp(0.0, 2.0)
            ride_feats[..., RIDE_FEAT_WAIT_VS_MEAN] = (wait - mean_wait).clamp(-1.0, 1.0)
            new_ride_emb = self._encode_rides(ride_feats)
            new_ride_emb = new_ride_emb + self.cross_attn.ride_guest_proj(
                guest_ctx
            ).unsqueeze(1)
            ride_ctx = torch.where(is_ride.view(batch, 1, 1), new_ride_emb, ride_ctx)
            cand_emb = ride_ctx.gather(
                1, cand_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
            )

            active = is_ride

        routes_out = torch.stack(route_list, dim=1)
        if routes is not None:
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
            stage_a_logits=slot0_logits,
            stage_a_mask=slot0_mask,
        )

    @staticmethod
    def _sample_or_argmax(
        logits: torch.Tensor,
        dist: torch.distributions.Categorical,
        *,
        deterministic: bool,
        close_margin: float,
        top_p: float,
    ) -> torch.Tensor:
        """Argmax, or sample when top-2 probs within close_margin (inference diversity)."""
        greedy = logits.argmax(dim=-1)
        if deterministic and close_margin <= 0:
            return greedy
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sampled = _nucleus_sample(probs, top_p)
        else:
            sampled = dist.sample()
        if not deterministic:
            return sampled
        top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
        if top2.size(-1) < 2:
            return greedy
        gap = top2[:, 0] - top2[:, 1]
        should_sample = gap < close_margin
        return torch.where(should_sample, sampled, greedy)


def _nucleus_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sample from nucleus (top-p) distribution. probs: (B, A)."""
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum <= top_p
    mask[..., 0] = True
    filtered = sorted_probs * mask.to(sorted_probs.dtype)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sample_pos = torch.distributions.Categorical(probs=filtered).sample()
    return sorted_idx.gather(1, sample_pos.unsqueeze(1)).squeeze(1)


# Back-compat alias used throughout training / companion / router.
ParkRouterModel = RankRouteModel


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
    model: RankRouteModel,
    guest: torch.Tensor,
    ride: torch.Tensor,
    env: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Stage A and apply legal-action masking (already applied inside)."""
    logits, values = model(guest, ride, env)
    mask = build_action_mask(guest, ride, env)
    return apply_action_mask(logits, mask), values, mask


def forward_route_with_mask(
    model: RankRouteModel,
    guest: torch.Tensor,
    ride: torch.Tensor,
    env: torch.Tensor,
    routes: torch.Tensor | None = None,
    *,
    deterministic: bool = False,
    force_first: torch.Tensor | None = None,
    temperature: float = 1.0,
    close_margin: float = 0.0,
    top_p: float = 1.0,
) -> RouteOutput:
    """Full rank-then-route decode (masking applied inside the model)."""
    return model.forward_route(
        guest,
        ride,
        env,
        routes=routes,
        deterministic=deterministic,
        force_first=force_first,
        temperature=temperature,
        close_margin=close_margin,
        top_p=top_p,
    )
