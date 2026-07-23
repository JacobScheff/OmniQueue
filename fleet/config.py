"""Fleet configuration: feature layouts, action space, and model hyperparameters.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Padded decision sizes (action / obs contract)
# ---------------------------------------------------------------------------

# Max free vehicles in one coordinated wave (padded).
MAX_VEHICLES = 64
# Max pending request candidates pointed at in one decision (padded).
MAX_REQUESTS = 128
# Special actions after request slots: STAY, IDLE (no chargers in v1).
NUM_SPECIAL_ACTIONS = 2
ACTION_STAY = MAX_REQUESTS  # index of STAY
ACTION_IDLE = MAX_REQUESTS + 1  # index of IDLE
NUM_ACTIONS = MAX_REQUESTS + NUM_SPECIAL_ACTIONS

# ---------------------------------------------------------------------------
# Feature dimensions (must match obs adapters / future C++ build_observation)
# ---------------------------------------------------------------------------

# Vehicle dynamic features (location via node index + optional GNN gather):
# 0 time_to_free (normalized), 1 capacity_frac, 2 soc, 3 busy_flag, 4–7 reserved
VEHICLE_FEAT_DIM = 8

# Request dynamic features (origin/dest via node indices + optional GNN gather):
# 0 wait_norm, 1 size_norm, 2 trip_length_norm, 3 deadline_norm,
# 4 available (>0.5 = still open), 5–7 reserved
REQUEST_FEAT_DIM = 8

# Pairwise vehicle ↔ request (from A* / cached shortest paths):
# 0 drive_time_norm, 1 distance_norm, 2 energy_norm, 3 reachable (>0.5)
PAIRWISE_FEAT_DIM = 4

# Global env: 0 time_of_day, 1 backlog_norm, 2 fleet_soc_mean, 3 free_vehicle_frac
ENV_FEAT_DIM = 4

# Node features for optional GNN (size-independent; no all-pairs distances):
# 0 x_norm, 1 y_norm, 2 n_requests_norm, 3 total_wait_norm
NODE_FEAT_DIM = 4

# Edge features: 0 length_norm, 1 travel_time_norm
EDGE_FEAT_DIM = 2

# Flat obs helper for single-vehicle (K=1) debugging — not the primary API.
FLAT_OBS_DIM = (
    VEHICLE_FEAT_DIM
    + MAX_REQUESTS * (REQUEST_FEAT_DIM + PAIRWISE_FEAT_DIM)
    + ENV_FEAT_DIM
)

# Feature column indices used by masking
REQUEST_FEAT_AVAILABLE = 4
PAIRWISE_FEAT_REACHABLE = 3

# ---------------------------------------------------------------------------
# Model architecture defaults
# ---------------------------------------------------------------------------

D_MODEL = 256
NUM_TRANSFORMER_LAYERS = 8
NUM_ATTN_HEADS = 8
NUM_GNN_LAYERS = 8
USE_GNN = True
# Upper bound for optional node-id embedding fallback when GNN is disabled.
MAX_NODES = 4096

def build_action_mask(
    request_features: torch.Tensor,
    pairwise_features: torch.Tensor,
    request_padding_mask: torch.Tensor | None = None,
    vehicle_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return boolean mask (B, K, A) where True = legal action.

    Rules (v1, no chargers):
    - request slot legal if padded-valid, available, and reachable from that vehicle
    - STAY always legal for valid vehicles
    - IDLE always legal for valid vehicles
    - padded vehicles get an all-False action mask (ignored by CE / sampling helpers)
    """

    if request_features.dim() == 2:
        request_features = request_features.unsqueeze(0)
    if pairwise_features.dim() == 3:
        pairwise_features = pairwise_features.unsqueeze(0)

    batch, num_vehicles, num_requests, _ = pairwise_features.shape
    device = pairwise_features.device

    if request_features.dim() == 3:
        # (B, R, F) → broadcast availability across vehicles
        available = request_features[..., REQUEST_FEAT_AVAILABLE] > 0.5  # (B, R)
        available = available.unsqueeze(1).expand(-1, num_vehicles, -1)
    else:
        available = request_features[..., REQUEST_FEAT_AVAILABLE] > 0.5  # (B, K, R)

    reachable = pairwise_features[..., PAIRWISE_FEAT_REACHABLE] > 0.5
    request_ok = available & reachable

    if request_padding_mask is not None:
        req_pad = request_padding_mask
        if req_pad.dim() == 1:
            req_pad = req_pad.unsqueeze(0)
        request_ok = request_ok & req_pad.unsqueeze(1)

    mask = torch.zeros(
        batch, num_vehicles, NUM_ACTIONS, dtype=torch.bool, device=device
    )
    mask[:, :, :num_requests] = request_ok
    mask[:, :, ACTION_STAY] = True
    mask[:, :, ACTION_IDLE] = True

    if vehicle_padding_mask is not None:
        veh_pad = vehicle_padding_mask
        if veh_pad.dim() == 1:
            veh_pad = veh_pad.unsqueeze(0)
        mask = mask & veh_pad.unsqueeze(-1)

    return mask


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal action logits to a large negative value (CE/Categorical-safe)."""

    return logits.masked_fill(
        ~mask, torch.tensor(-1.0e9, device=logits.device, dtype=logits.dtype)
    )


def masked_cross_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    vehicle_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy with illegal actions masked; ignores padded vehicles.

    logits: (B, K, A), actions: (B, K), action_mask: (B, K, A)
    vehicle_padding_mask: (B, K) True = valid vehicle
    """

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_actions = actions.reshape(-1)
    flat_mask = action_mask.reshape(-1, action_mask.size(-1))

    if vehicle_padding_mask is not None:
        valid = vehicle_padding_mask.reshape(-1).to(dtype=torch.bool)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        flat_logits = flat_logits[valid]
        flat_actions = flat_actions[valid]
        flat_mask = flat_mask[valid]

    flat_mask = flat_mask.clone()
    flat_mask.scatter_(
        1,
        flat_actions.clamp(min=0, max=flat_mask.size(-1) - 1).unsqueeze(1),
        True,
    )

    # Guarantee ≥1 legal logit (STAY) if a row was fully masked.
    none_legal = ~flat_mask.any(dim=-1)
    flat_mask[none_legal, ACTION_STAY] = True

    masked_logits = apply_action_mask(flat_logits, flat_mask)
    return F.cross_entropy(masked_logits, flat_actions)
