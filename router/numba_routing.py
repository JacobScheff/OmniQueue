"""Batch routing kernel with optional Numba acceleration."""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def has_numba() -> bool:
    return _HAS_NUMBA


def _route_batch_impl(
    party_ids: np.ndarray,
    leave_sec: np.ndarray,
    location_node_idx: np.ndarray,
    effective_speed: np.ndarray,
    preference_order: np.ndarray,
    balk_sec: np.ndarray,
    node_idx_to_ride: np.ndarray,
    open_mask: np.ndarray,
    wait_times: np.ndarray,
    durations: np.ndarray,
    base_walk_to_rides: np.ndarray,
    base_walking_speed: float,
    now_sec: int,
    rand_u01: np.ndarray,
    idle_prob: float,
    exit_code: int,
    idle_code: int,
) -> np.ndarray:
    batch_size = party_ids.shape[0]
    targets = np.empty(batch_size, dtype=np.int32)
    num_rides = open_mask.shape[0]

    for b in range(batch_size):
        pid = int(party_ids[b])
        if now_sec >= leave_sec[pid]:
            targets[b] = exit_code
            continue

        remaining = int(leave_sec[pid]) - now_sec
        node_idx = int(location_node_idx[pid])
        speed = float(effective_speed[pid])
        if speed < 0.1:
            speed = 0.1
        scale = base_walking_speed / speed

        current_ride = int(node_idx_to_ride[node_idx])
        chosen = -1

        for k in range(num_rides):
            ride_id = int(preference_order[pid, k])
            if current_ride >= 0 and ride_id == current_ride:
                continue
            if not open_mask[ride_id]:
                continue
            base_walk = int(base_walk_to_rides[node_idx, ride_id])
            walk = max(1, int(math.ceil(base_walk * scale)))
            if walk + wait_times[ride_id] + durations[ride_id] > remaining:
                continue
            if wait_times[ride_id] <= balk_sec[pid, ride_id]:
                chosen = ride_id
                break

        if chosen >= 0:
            targets[b] = chosen
            continue

        if rand_u01[b] < idle_prob:
            targets[b] = idle_code
            continue

        forced = exit_code
        for k in range(num_rides):
            ride_id = int(preference_order[pid, k])
            if current_ride >= 0 and ride_id == current_ride:
                continue
            if not open_mask[ride_id]:
                continue
            base_walk = int(base_walk_to_rides[node_idx, ride_id])
            walk = max(1, int(math.ceil(base_walk * scale)))
            if walk + wait_times[ride_id] + durations[ride_id] > remaining:
                continue
            forced = ride_id
            break

        targets[b] = forced

    return targets


if _HAS_NUMBA:
    route_batch_numba = njit(cache=True)(_route_batch_impl)
else:
    route_batch_numba = _route_batch_impl
