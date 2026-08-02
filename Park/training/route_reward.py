"""Python-side walk shaping for multi-ride rank-then-route plans."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import Park.config as config
from Park.training.features import NUM_RIDES, RIDE_FEAT_WALK

ROUTE_PAD = -1

_ride_walk_sec: np.ndarray | None = None


def route_k() -> int:
    return int(getattr(config, "PPO_ROUTE_K", 5))


def ride_walk_matrix_sec() -> np.ndarray:
    """Cached (R, R) walk seconds between ride nodes at BASE_WALKING_SPEED."""
    global _ride_walk_sec
    if _ride_walk_sec is None:
        from Park.park_graph import ParkGraph

        graph = ParkGraph()
        mat = np.zeros((NUM_RIDES, NUM_RIDES), dtype=np.float32)
        speed = float(config.BASE_WALKING_SPEED)
        for i in range(NUM_RIDES):
            mat[i] = graph.walk_times_to_rides(graph.ride_node(i), speed).astype(np.float32)
        _ride_walk_sec = mat
    return _ride_walk_sec


def is_ride_action(action: int) -> bool:
    return 0 <= int(action) < NUM_RIDES


def pad_route(actions: Sequence[int], k: int | None = None) -> np.ndarray:
    """Pack a route into length-K with ROUTE_PAD after exit/idle or short lists."""
    k = route_k() if k is None else int(k)
    out = np.full(k, ROUTE_PAD, dtype=np.int64)
    if not actions:
        return out
    a0 = int(actions[0])
    out[0] = a0
    if not is_ride_action(a0):
        return out
    n = min(len(actions), k)
    for i in range(1, n):
        a = int(actions[i])
        if not is_ride_action(a):
            break
        out[i] = a
    return out


def commit_action(route: np.ndarray | Sequence[int]) -> int:
    return int(route[0])


def planned_walk_penalty(route: np.ndarray) -> float:
    """Mean inter-ride walk along the emitted ride route, normalized."""
    route = np.asarray(route, dtype=np.int64).reshape(-1)
    if not is_ride_action(int(route[0])):
        return 0.0
    rides = [int(a) for a in route.tolist() if is_ride_action(int(a))]
    if len(rides) < 2:
        return 0.0
    mat = ride_walk_matrix_sec()
    total = 0.0
    hops = 0
    for i in range(len(rides) - 1):
        total += float(mat[rides[i], rides[i + 1]])
        hops += 1
    norm = float(getattr(config, "PPO_ROUTE_WALK_NORM_SEC", 600.0))
    coef = float(getattr(config, "PPO_ROUTE_PLANNED_WALK_COEF", 0.01))
    return coef * ((total / max(hops, 1)) / max(norm, 1.0))


def realized_walk_penalty(walk_sec: float) -> float:
    if walk_sec <= 0.0:
        return 0.0
    norm = float(getattr(config, "PPO_ROUTE_WALK_NORM_SEC", 600.0))
    coef = float(getattr(config, "PPO_ROUTE_REALIZED_WALK_COEF", 0.02))
    return coef * (float(walk_sec) / max(norm, 1.0))


def walk_sec_to_commit(ride_feats: np.ndarray, commit: int) -> float:
    """Walk seconds to committed ride from current obs ride features (feat 5)."""
    if not is_ride_action(commit):
        return 0.0
    walk_norm = float(ride_feats[commit, RIDE_FEAT_WALK])
    return max(0.0, walk_norm) * 3600.0


def route_shaping_delta(
    new_route: np.ndarray,
    prev_route: np.ndarray | None,
    ride_feats: np.ndarray,
) -> tuple[float, float]:
    """Return (emit_shaping, realized_walk_sec_to_apply_later).

    emit_shaping = -planned_walk (applied when reward for this transition arrives).
    ``prev_route`` is accepted for API stability but unused (consistency removed).
    realized_walk_sec is stored and converted with ``realized_walk_penalty`` later.
    """
    del prev_route  # consistency shaping removed in rank_route_v1
    planned = planned_walk_penalty(new_route)
    commit = commit_action(new_route)
    walk_sec = walk_sec_to_commit(ride_feats, commit)
    return -planned, walk_sec
