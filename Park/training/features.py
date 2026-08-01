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
