"""Feature dimensions shared with the C++ simulator and ParkRouterModel."""

from __future__ import annotations

GUEST_FEAT_DIM = 45
RIDE_DYNAMIC_FEAT_DIM = 5
ENV_DYNAMIC_FEAT_DIM = 4
NUM_RIDES = 35
NUM_ACTIONS = 37  # 35 rides + exit + idle
FLAT_OBS_DIM = GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM + ENV_DYNAMIC_FEAT_DIM
