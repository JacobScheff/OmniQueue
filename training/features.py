"""Feature dimensions shared with the C++ simulator and ParkRouterModel."""

from __future__ import annotations

import torch
import torch.nn.functional as F

GUEST_FEAT_DIM = 45
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

# Guest feature indices used for masking
GUEST_FEAT_TIME_LEFT = 37
GUEST_FEAT_AT_RIDE_NODE = 41


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
    """Set illegal action logits to a large negative value."""
    neg = torch.finfo(logits.dtype).min / 2
    return logits.masked_fill(~mask, neg)


def masked_cross_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    guest_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy with illegal actions masked; optional pad over guest axis.

    logits: (B, G, A), actions: (B, G), action_mask: (B, G, A)
    guest_padding_mask: (B, G) True = valid guest
    """
    masked_logits = apply_action_mask(logits, action_mask)
    flat_logits = masked_logits.reshape(-1, masked_logits.size(-1))
    flat_actions = actions.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_actions, reduction="none")
    loss = loss.view_as(actions)
    if guest_padding_mask is None:
        return loss.mean()
    weights = guest_padding_mask.to(dtype=loss.dtype)
    denom = weights.sum().clamp(min=1.0)
    return (loss * weights).sum() / denom
