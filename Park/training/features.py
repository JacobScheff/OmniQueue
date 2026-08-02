"""Feature dimensions shared with the C++ simulator and ParkRouterModel.

Constants are importable without torch (needed by the ONNX companion image).
Torch is only required for the masking helpers used in training / torch inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

GUEST_FEAT_DIM = 46
# 0..33 preferences, 34 remaining sharpened pref mass (Σ pref**EXP unfinished),
# 35..44 party state, 45 elapsed_since_spawn
# 0 wait, 1 incoming, 2 open, 3 duration, 4 capacity, 5 walk, 6 history, 7 must_do,
# 8 unfinished sharpened pref (0 if already ridden)
RIDE_DYNAMIC_FEAT_DIM = 9
# Legacy companion ONNX (v1/v2) was trained with 8 ride feats (no pref column).
RIDE_DYNAMIC_FEAT_DIM_LEGACY = 8
ENV_DYNAMIC_FEAT_DIM = 4
NUM_RIDES = 34
NUM_ACTIONS = 36  # 34 rides + exit + idle
FLAT_OBS_DIM = GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM + ENV_DYNAMIC_FEAT_DIM

# Model architecture defaults
D_MODEL = 256
ROUTE_K = 6  # default; overridden by config.PPO_ROUTE_K when available
ROUTE_PAD = -1
ACTION_EXIT = NUM_RIDES
ACTION_IDLE = NUM_RIDES + 1

DAY_SECONDS = 54000.0
CLOSE_DRAIN_SEC = 3.0 * 3600.0

# Ride feature column indices (must match native build_observation)
RIDE_FEAT_WAIT = 0
RIDE_FEAT_OPEN = 2
RIDE_FEAT_DURATION = 3
RIDE_FEAT_WALK = 5
RIDE_FEAT_HISTORY = 6
RIDE_FEAT_MUST_DO = 7
RIDE_FEAT_UNFINISHED_PREF = 8

# Guest feature indices used for masking / diagnostics
GUEST_FEAT_REMAINING_PREF_MASS = 34
GUEST_FEAT_TIME_LEFT = 37
GUEST_FEAT_MUST_DO_COUNT = 40
GUEST_FEAT_AT_RIDE_NODE = 41
GUEST_FEAT_ELAPSED_SINCE_SPAWN = 45


def route_k() -> int:
    try:
        import Park.config as config

        return int(getattr(config, "PPO_ROUTE_K", ROUTE_K))
    except Exception:
        return ROUTE_K


def build_action_mask(
    guest: torch.Tensor,
    ride: torch.Tensor,
    env: torch.Tensor,
) -> torch.Tensor:
    """Return boolean mask (B, A) where True = legal action.

    Rules mirror the heuristic router feasibility / soft-close policy:
    - after official close (env time >= 1) or no time left → exit only
    - closed rides masked
    - ride the party is already at (walk == 0 while at a ride node) masked
    - rides whose walk+wait+duration exceed remaining park time masked
    - idle masked after soft close; exit always allowed
    """
    import torch

    batch, num_rides, _ = ride.shape
    device = ride.device

    open_ok = ride[..., RIDE_FEAT_OPEN] > 0.5
    walk = ride[..., RIDE_FEAT_WALK].clamp(min=0.0) * 3600.0
    wait = ride[..., RIDE_FEAT_WAIT].clamp(min=0.0) * 3600.0
    duration = ride[..., RIDE_FEAT_DURATION].clamp(min=0.0) * 900.0

    time_left_frac = guest[..., GUEST_FEAT_TIME_LEFT].clamp(min=0.0)
    remaining_sec = time_left_frac * DAY_SECONDS
    day_frac = env[..., 0]
    soft_closed = (day_frac >= 1.0) | (time_left_frac <= 0.0)

    drain = torch.where(
        day_frac < 1.0,
        torch.full((batch,), CLOSE_DRAIN_SEC, device=device, dtype=ride.dtype),
        torch.zeros(batch, device=device, dtype=ride.dtype),
    )
    remaining_for_feas = (remaining_sec + drain).unsqueeze(-1)
    time_ok = (walk + wait + duration) <= remaining_for_feas

    at_ride_node = guest[..., GUEST_FEAT_AT_RIDE_NODE] > 0.5
    already_here = at_ride_node.unsqueeze(-1) & (ride[..., RIDE_FEAT_WALK] <= 1e-6)

    ride_ok = open_ok & time_ok & (~already_here) & (~soft_closed.unsqueeze(-1))

    mask = torch.zeros(batch, NUM_ACTIONS, dtype=torch.bool, device=device)
    mask[:, :num_rides] = ride_ok
    mask[:, NUM_RIDES] = True
    mask[:, NUM_RIDES + 1] = ~soft_closed
    return mask


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal action logits to a large negative value (finite, CE/Categorical-safe)."""
    import torch

    return logits.masked_fill(~mask, torch.tensor(-1.0e9, device=logits.device, dtype=logits.dtype))


def masked_cross_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy with illegal actions masked.

    logits: (B, A), actions: (B,), action_mask: (B, A)
    """
    import torch.nn.functional as F

    mask = action_mask.clone()
    mask.scatter_(1, actions.clamp(min=0, max=mask.size(-1) - 1).unsqueeze(1), True)

    if mask.size(-1) > NUM_RIDES:
        none_legal = ~mask.any(dim=-1)
        mask[none_legal, NUM_RIDES] = True

    masked_logits = apply_action_mask(logits, mask)
    return F.cross_entropy(masked_logits, actions)


def build_tail_ride_mask(
    ride: torch.Tensor,
    picked: torch.Tensor,
) -> torch.Tensor:
    """Legal rides for route slots k>=1: open, unfinished, not already picked.

    ride: (B, R, F), picked: (B, R) bool
    returns (B, R) bool
    """
    import torch

    open_ok = ride[..., RIDE_FEAT_OPEN] > 0.5
    unfinished = ride[..., RIDE_FEAT_HISTORY] <= 0.5
    return open_ok & unfinished & (~picked)


def rewrite_prefs_must_dos(
    guest: torch.Tensor,
    ride: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    must_do_boost: float = 10.0,
    pref_eps: float = 1e-3,
    max_must_dos: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clone obs tensors with freshly sampled training-style prefs/must-dos.

    Waits, walks, history, time, and location are left unchanged.
    """
    import torch

    try:
        import Park.config as config

        must_do_boost = float(getattr(config, "MUST_DO_PREF_BOOST", must_do_boost))
        pref_eps = float(getattr(config, "PREF_RAW_EPS", pref_eps))
    except Exception:
        pass

    batch, num_rides, _ = ride.shape
    device = guest.device
    dtype = guest.dtype
    new_guest = guest.clone()
    new_ride = ride.clone()
    history = ride[..., RIDE_FEAT_HISTORY] > 0.5

    for b in range(batch):
        raw = torch.empty(num_rides, device=device, dtype=dtype)
        if generator is None:
            raw.uniform_(pref_eps, 1.0)
            n_must = int(torch.randint(0, max_must_dos + 1, (1,)).item())
        else:
            raw.uniform_(pref_eps, 1.0, generator=generator)
            n_must = int(
                torch.randint(0, max_must_dos + 1, (1,), generator=generator).item()
            )
        avail = (~history[b]).nonzero(as_tuple=False).flatten()
        must_ids: list[int] = []
        if avail.numel() > 0 and n_must > 0:
            n_take = min(n_must, int(avail.numel()))
            if generator is None:
                perm = torch.randperm(avail.numel(), device=device)[:n_take]
            else:
                perm = torch.randperm(avail.numel(), device=device, generator=generator)[
                    :n_take
                ]
            must_ids = [int(avail[i].item()) for i in perm.tolist()]
            for mid in must_ids:
                raw[mid] = raw[mid] * must_do_boost
        prefs = raw / raw.sum().clamp(min=1e-8)
        new_guest[b, :num_rides] = prefs
        pref_exp = float(getattr(config, "PPO_PREF_REWARD_EXP", 1.0))
        sharpened = prefs.clamp(min=0.0).pow(pref_exp)
        rem = (sharpened * (~history[b]).to(dtype)).sum()
        new_guest[b, GUEST_FEAT_REMAINING_PREF_MASS] = rem
        new_guest[b, GUEST_FEAT_MUST_DO_COUNT] = float(len(must_ids)) / 5.0
        new_ride[b, :, RIDE_FEAT_MUST_DO] = 0.0
        for mid in must_ids:
            new_ride[b, mid, RIDE_FEAT_MUST_DO] = 1.0
        if new_ride.size(-1) > RIDE_FEAT_UNFINISHED_PREF:
            new_ride[b, :, RIDE_FEAT_UNFINISHED_PREF] = sharpened * (
                ~history[b]
            ).to(dtype)
    return new_guest, new_ride


def top_must_do_or_pref(guest: torch.Tensor, ride: torch.Tensor) -> torch.Tensor:
    """Per-batch top unfinished must-do, else top unfinished preference ride."""
    import torch

    batch, num_rides, _ = ride.shape
    unfinished = ride[..., RIDE_FEAT_HISTORY] <= 0.5
    must = (ride[..., RIDE_FEAT_MUST_DO] > 0.5) & unfinished
    prefs = guest[:, :num_rides].clone()
    prefs = prefs.masked_fill(~unfinished, -1.0)
    out = torch.empty(batch, dtype=torch.long, device=guest.device)
    for b in range(batch):
        must_idx = must[b].nonzero(as_tuple=False).flatten()
        if must_idx.numel() > 0:
            scores = prefs[b, must_idx]
            out[b] = must_idx[scores.argmax()]
        else:
            out[b] = prefs[b].argmax()
    return out


def pref_rank_aux_loss(
    slot_logits: "torch.Tensor",
    slot_masks: "torch.Tensor",
    ride: "torch.Tensor",
    *,
    slots: tuple[int, ...] = (1, 2),
    must_do_bonus: float = 0.5,
) -> "torch.Tensor":
    """Soft CE encouraging early tail slots toward unfinished high-pref rides.

    slot_logits / slot_masks: (B, K, A); ride: (B, R, F).
    Uses unfinished sharpened pref (feat 8) plus a must-do bonus as soft targets,
    renormalized under each slot's legal ride mask. Returns a scalar (0 if no
    valid rows).
    """
    import torch
    import torch.nn.functional as F

    try:
        import Park.config as config

        slots = tuple(getattr(config, "PPO_PREF_RANK_SLOTS", slots))
        must_do_bonus = float(
            getattr(config, "PPO_PREF_RANK_MUST_DO_BONUS", must_do_bonus)
        )
    except Exception:
        pass

    prefs = ride[..., RIDE_FEAT_UNFINISHED_PREF].clamp(min=0.0)
    must = (ride[..., RIDE_FEAT_MUST_DO] > 0.5).to(prefs.dtype)
    scores = prefs + must_do_bonus * must

    slot_losses: list[torch.Tensor] = []
    k_max = int(slot_logits.size(1))
    for s in slots:
        if s <= 0 or s >= k_max:
            continue
        logits = slot_logits[:, s, :NUM_RIDES]
        mask = slot_masks[:, s, :NUM_RIDES]
        legal_scores = scores * mask.to(scores.dtype)
        mass = legal_scores.sum(dim=-1)
        valid = (mass > 1e-8) & mask.any(dim=-1)
        if not bool(valid.any()):
            continue
        target = legal_scores / mass.clamp(min=1e-8).unsqueeze(-1)
        log_probs = F.log_softmax(apply_action_mask(logits, mask), dim=-1)
        ce = -(target * log_probs).sum(dim=-1)
        slot_losses.append(ce[valid].mean())

    if not slot_losses:
        return slot_logits.new_zeros(())
    return torch.stack(slot_losses).mean()


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen–Shannon divergence for rows of categorical probs. Returns (B,)."""
    import torch

    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm
