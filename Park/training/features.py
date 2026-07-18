"""Feature dimensions shared with the C++ simulator and ParkRouterModel.

Constants are importable without torch (needed by the ONNX companion image).
Torch is only required for the masking helpers used in training / torch inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

GUEST_FEAT_DIM = 46
# 0..33 preferences, 34 remaining_pref_mass, 35..44 party state, 45 elapsed_since_spawn
# 0 wait, 1 incoming, 2 open, 3 duration, 4 capacity, 5 walk, 6 history, 7 must_do
RIDE_DYNAMIC_FEAT_DIM = 8
ENV_DYNAMIC_FEAT_DIM = 4
NUM_RIDES = 34
NUM_ACTIONS = 36  # 34 rides + exit + idle
FLAT_OBS_DIM = GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM + ENV_DYNAMIC_FEAT_DIM

# Model architecture defaults
D_MODEL = 256
NUM_TRANSFORMER_LAYERS = 3
NUM_ATTN_HEADS = 8

DAY_SECONDS = 54000.0
CLOSE_DRAIN_SEC = 3.0 * 3600.0

# Ride feature column indices (must match native build_observation)
RIDE_FEAT_WAIT = 0
RIDE_FEAT_OPEN = 2
RIDE_FEAT_DURATION = 3
RIDE_FEAT_WALK = 5

# Guest feature indices used for masking / diagnostics
GUEST_FEAT_REMAINING_PREF_MASS = 34
GUEST_FEAT_TIME_LEFT = 37
GUEST_FEAT_AT_RIDE_NODE = 41
GUEST_FEAT_ELAPSED_SINCE_SPAWN = 45


def build_action_mask(
    guest: torch.Tensor,
    ride: torch.Tensor,
    env: torch.Tensor,
) -> torch.Tensor:
    """Return boolean mask (B, G, A) where True = legal action.

    Rules mirror the heuristic router feasibility / soft-close policy:
    - after official close (env time >= 1) or no time left → exit only
    - closed rides masked
    - ride the party is already at (walk == 0 while at a ride node) masked
    - rides whose walk+wait+duration exceed remaining park time masked
    - idle masked after soft close; exit always allowed
    """
    import torch

    if ride.dim() == 3:
        ride = ride.unsqueeze(1)
    if guest.dim() == 2:
        guest = guest.unsqueeze(1)

    batch, num_guests, num_rides, _ = ride.shape
    device = ride.device

    open_ok = ride[..., RIDE_FEAT_OPEN] > 0.5
    walk = ride[..., RIDE_FEAT_WALK].clamp(min=0.0) * 3600.0
    wait = ride[..., RIDE_FEAT_WAIT].clamp(min=0.0) * 3600.0
    duration = ride[..., RIDE_FEAT_DURATION].clamp(min=0.0) * 900.0

    time_left_frac = guest[..., GUEST_FEAT_TIME_LEFT].clamp(min=0.0)
    remaining_sec = time_left_frac * DAY_SECONDS
    day_frac = env[..., 0].view(batch, 1).expand(batch, num_guests)
    soft_closed = (day_frac >= 1.0) | (time_left_frac <= 0.0)

    # Approximate C++ drain window for parties staying until official close.
    drain = torch.where(
        day_frac < 1.0,
        torch.full((batch, num_guests), CLOSE_DRAIN_SEC, device=device, dtype=ride.dtype),
        torch.zeros(batch, num_guests, device=device, dtype=ride.dtype),
    )
    remaining_for_feas = (remaining_sec + drain).unsqueeze(-1)
    time_ok = (walk + wait + duration) <= remaining_for_feas

    at_ride_node = guest[..., GUEST_FEAT_AT_RIDE_NODE] > 0.5
    already_here = at_ride_node.unsqueeze(-1) & (ride[..., RIDE_FEAT_WALK] <= 1e-6)

    ride_ok = open_ok & time_ok & (~already_here) & (~soft_closed.unsqueeze(-1))

    mask = torch.zeros(batch, num_guests, NUM_ACTIONS, dtype=torch.bool, device=device)
    mask[:, :, :num_rides] = ride_ok
    mask[:, :, NUM_RIDES] = True
    mask[:, :, NUM_RIDES + 1] = ~soft_closed
    return mask


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal action logits to a large negative value (finite, CE/Categorical-safe)."""
    import torch

    # Avoid dtype.min: float32 min/2 still blows up softmax/CE; -1e9 is standard.
    return logits.masked_fill(~mask, torch.tensor(-1.0e9, device=logits.device, dtype=logits.dtype))


def masked_cross_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    guest_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy with illegal actions masked; ignores padded guests.

    logits: (B, G, A), actions: (B, G), action_mask: (B, G, A)
    guest_padding_mask: (B, G) True = valid guest

    Important: padded rows must not be scored. Masking all actions to -inf and then
    multiplying CE by 0 yields ``0 * inf = nan`` and poisons the epoch loss.
    """
    import torch.nn.functional as F

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_actions = actions.reshape(-1)
    flat_mask = action_mask.reshape(-1, action_mask.size(-1))

    if guest_padding_mask is not None:
        valid = guest_padding_mask.reshape(-1).to(dtype=torch.bool)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        flat_logits = flat_logits[valid]
        flat_actions = flat_actions[valid]
        flat_mask = flat_mask[valid]

    # Always keep the supervised label unmasked so a slightly-strict feasibility
    # mask cannot send CE to +inf on an expert action.
    flat_mask = flat_mask.clone()
    flat_mask.scatter_(1, flat_actions.clamp(min=0, max=flat_mask.size(-1) - 1).unsqueeze(1), True)

    # Guarantee ≥1 legal logit per row (exit) if a row was fully masked.
    if flat_mask.size(-1) > NUM_RIDES:
        none_legal = ~flat_mask.any(dim=-1)
        flat_mask[none_legal, NUM_RIDES] = True

    masked_logits = apply_action_mask(flat_logits, flat_mask)
    return F.cross_entropy(masked_logits, flat_actions)
