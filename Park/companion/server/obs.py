"""Build ParkRouterModel observations from live waits + user state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Park import config
from Park.companion.server.waits import LiveBoard
from Park.park_graph import get_park_graph
from Park.training.features import (
    ENV_DYNAMIC_FEAT_DIM,
    FLAT_OBS_DIM,
    GUEST_FEAT_DIM,
    NUM_ACTIONS,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
)

# PartyState.Walking — typical “about to choose next ride” pose for live use.
_STATE_WALKING = 1.0
WEIGHT_SLIDER_MAX = 250.0


@dataclass
class CompanionState:
    """User-editable companion inputs (mirrors what the phone stores)."""

    preference_weights: np.ndarray  # raw weights, length NUM_RIDES
    must_dos: np.ndarray  # 0/1 uint8, length NUM_RIDES
    history: np.ndarray  # completion counts, length NUM_RIDES
    location_node_id: int  # park graph node id (hub or ride node)
    leave_sec: int | None = None  # seconds since park open; None → stay until close
    spawn_sec: int = 0  # arrival time since open
    party_size: int = 2
    walking_speed: float = float(config.BASE_WALKING_SPEED)


def normalize_preferences(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32).copy()
    if w.shape != (NUM_RIDES,):
        raise ValueError(f"preference_weights must have shape ({NUM_RIDES},), got {w.shape}")
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 1e-8:
        w[:] = 1.0 / NUM_RIDES
    else:
        w /= total
    return w


def default_preference_weights() -> np.ndarray:
    pops = np.array([float(r["popularity"]) for r in config.RIDES], dtype=np.float32)
    return np.clip(pops, 0.0, WEIGHT_SLIDER_MAX)


def compute_balk_sec(prefs: np.ndarray) -> np.ndarray:
    return (
        config.BASE_BALK_SEC
        + config.BALK_SCALE * np.power(np.asarray(prefs, dtype=np.float32), config.BALK_PREF_EXP)
    ).astype(np.float32)


def now_sec_of_day(*, hour: int | None = None, minute: int | None = None) -> int:
    """Seconds since park open (clamped to the operating window)."""
    import datetime as dt

    if hour is None or minute is None:
        # Disneyland local time
        try:
            from zoneinfo import ZoneInfo

            now = dt.datetime.now(ZoneInfo("America/Los_Angeles"))
        except Exception:  # noqa: BLE001
            now = dt.datetime.utcnow()
        hour, minute = now.hour, now.minute
    abs_sec = hour * 3600 + minute * 60
    open_sec = config.DAY_START_HOUR * 3600
    rel = abs_sec - open_sec
    return int(np.clip(rel, 0, config.DAY_SECONDS))


def resolve_location_node_id(location_key: str) -> int:
    """Parse location keys like 'entrance', 'hub:1', 'ride:12'."""
    key = location_key.strip().lower()
    if key in ("entrance", "hub:0", f"node:{config.NODE_ENTRANCE}"):
        return config.NODE_ENTRANCE
    if key.startswith("hub:"):
        hub_id = int(key.split(":", 1)[1])
        if hub_id not in config.HUB_COORDS:
            raise ValueError(f"unknown hub id {hub_id}")
        return hub_id
    if key.startswith("ride:"):
        ride_id = int(key.split(":", 1)[1])
        if not (0 <= ride_id < NUM_RIDES):
            raise ValueError(f"ride id out of range: {ride_id}")
        return config.ride_node_id(ride_id)
    if key.startswith("node:"):
        return int(key.split(":", 1)[1])
    raise ValueError(f"invalid location key: {location_key!r}")


def build_live_observation(
    state: CompanionState,
    board: LiveBoard,
    *,
    now_sec: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Return (flat obs [FLAT_OBS_DIM], debug meta)."""
    park = get_park_graph()
    prefs = normalize_preferences(state.preference_weights)
    history = np.asarray(state.history, dtype=np.int32)
    if history.shape != (NUM_RIDES,):
        raise ValueError(f"history must have shape ({NUM_RIDES},)")
    must_flags = np.asarray(state.must_dos, dtype=np.uint8)
    if must_flags.shape != (NUM_RIDES,):
        raise ValueError(f"must_dos must have shape ({NUM_RIDES},)")
    must_remaining = ((must_flags > 0) & (history == 0)).astype(np.uint8)

    if now_sec is None:
        now_sec = now_sec_of_day()
    leave_sec = int(state.leave_sec) if state.leave_sec is not None else config.DAY_SECONDS
    leave_sec = max(leave_sec, now_sec)

    guest = np.zeros(GUEST_FEAT_DIM, dtype=np.float32)
    guest[0:NUM_RIDES] = prefs
    guest[34] = float(prefs[history == 0].sum())
    guest[35] = float(np.clip(state.party_size, 1, 16)) / 8.0
    guest[36] = float(state.walking_speed) / 2.0
    guest[37] = float(leave_sec - now_sec) / float(config.DAY_SECONDS)
    loc_idx = park.node_to_idx(state.location_node_id)
    guest[38] = float(loc_idx) / float(park.num_nodes)
    guest[39] = float(min(int(history.sum()), 40)) / 20.0
    guest[40] = float(must_remaining.sum()) / 5.0
    at_ride = int(park.node_idx_to_ride[loc_idx])
    guest[41] = 1.0 if at_ride >= 0 else 0.0
    guest[42] = _STATE_WALKING / 16.0
    balk = compute_balk_sec(prefs)
    guest[43] = float(balk.mean()) / 3600.0
    guest[44] = 0.0  # not mid-walk toward a target
    guest[45] = float(max(0, now_sec - int(state.spawn_sec))) / float(config.DAY_SECONDS)

    ride = np.zeros((NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM), dtype=np.float32)
    walk_secs = park.walk_times_to_rides(state.location_node_id, state.walking_speed)
    by_id = board.by_ride_id()
    open_count = 0
    wait_sum = 0.0
    wait_n = 0
    broken = 0
    warnings: list[str] = []

    for r in range(NUM_RIDES):
        live = by_id.get(r)
        wait_sec = 0.0
        is_open = False
        if live is not None:
            is_open = bool(live.open)
            if live.wait_min is not None:
                wait_sec = float(live.wait_min) * 60.0
            elif is_open:
                # Operating but no posted wait — treat as short/unknown
                wait_sec = 5.0 * 60.0
                warnings.append(f"no posted wait for {config.RIDES[r]['name']}; using 5 min")
        if not is_open:
            broken += 1
        else:
            open_count += 1
            wait_sum += wait_sec
            wait_n += 1

        ride[r, 0] = min(wait_sec, 3600.0) / 3600.0
        ride[r, 1] = 0.0  # incoming unavailable from public APIs
        ride[r, 2] = 1.0 if is_open else 0.0
        ride[r, 3] = float(config.RIDES[r]["duration_sec"]) / 900.0
        ride[r, 4] = float(config.RIDES[r]["capacity_per_hour"]) / 3600.0
        if at_ride == r:
            ride[r, 5] = 0.0
        else:
            ride[r, 5] = min(float(walk_secs[r]), 3600.0) / 3600.0
        ride[r, 6] = min(float(history[r]), 10.0) / 10.0
        ride[r, 7] = 1.0 if must_remaining[r] else 0.0

    warnings.append("incoming queue pressure set to 0 (not provided by wait API)")

    env = np.zeros(ENV_DYNAMIC_FEAT_DIM, dtype=np.float32)
    env[0] = float(now_sec) / float(config.DAY_SECONDS)
    mean_wait = (wait_sum / wait_n) if wait_n else 0.0
    env[1] = float(mean_wait / 3600.0)
    env[2] = 0.0
    env[3] = float(broken) / float(NUM_RIDES)

    flat = np.concatenate([guest, ride.reshape(-1), env]).astype(np.float32)
    assert flat.shape == (FLAT_OBS_DIM,), flat.shape

    meta = {
        "now_sec": now_sec,
        "leave_sec": leave_sec,
        "location_node_id": state.location_node_id,
        "location_node_idx": loc_idx,
        "at_ride_id": at_ride if at_ride >= 0 else None,
        "open_rides": open_count,
        "mean_wait_min": mean_wait / 60.0,
        "broken_fraction": float(broken) / float(NUM_RIDES),
        "preferences": prefs.tolist(),
        "must_remaining": must_remaining.astype(int).tolist(),
        "warnings": warnings,
        "board_error": board.error,
        "board_age_sec": None,
    }
    return flat, meta


ACTION_LABELS: list[str] = [r["name"] for r in config.RIDES] + ["Exit park", "Idle wander"]


def action_label(action_id: int) -> str:
    if 0 <= action_id < len(ACTION_LABELS):
        return ACTION_LABELS[action_id]
    return f"action_{action_id}"


def is_ride_action(action_id: int) -> bool:
    return 0 <= action_id < NUM_RIDES


__all__ = [
    "ACTION_LABELS",
    "CompanionState",
    "NUM_ACTIONS",
    "NUM_RIDES",
    "WEIGHT_SLIDER_MAX",
    "action_label",
    "build_live_observation",
    "default_preference_weights",
    "is_ride_action",
    "normalize_preferences",
    "now_sec_of_day",
    "resolve_location_node_id",
]
