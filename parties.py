"""Party pool with struct-of-arrays layout for fast routing."""

from __future__ import annotations

import numpy as np

import config
from park_graph import ParkGraph, get_park_graph
from park_types import Party, PartyState, ROUTE_IDLE_CODE


def _compute_balk_sec_array(preferences: np.ndarray) -> np.ndarray:
    return config.BASE_BALK_SEC + config.BALK_SCALE * (preferences ** config.BALK_PREF_EXP)


def _compute_preference_order_row(
    preferences: np.ndarray, must_do_remaining: np.ndarray
) -> np.ndarray:
    order = np.arange(config.NUM_RIDES, dtype=np.int16)
    must_keys = np.where(must_do_remaining, 0, 1)
    sort_idx = np.lexsort((order, -preferences, must_keys))
    return order[sort_idx].astype(np.int16)


def _party_effective_speed(rng: np.random.Generator, party_size: int) -> float:
    member_speeds = rng.lognormal(
        config.MEMBER_SPEED_LOG_MU, config.MEMBER_SPEED_LOG_SIGMA, size=party_size
    )
    return float(np.min(member_speeds))


class PartyPool:
    """Struct-of-arrays party storage; `get()` builds a Party view for tests."""

    def __init__(self, graph: ParkGraph | None = None) -> None:
        self.graph = graph or get_park_graph()
        self.count = 0
        self.party_size = np.empty(0, dtype=np.int32)
        self.spawn_sec = np.empty(0, dtype=np.int32)
        self.leave_sec = np.empty(0, dtype=np.int32)
        self.location_node_idx = np.empty(0, dtype=np.int32)
        self.effective_speed = np.empty(0, dtype=np.float32)
        self.state = np.empty(0, dtype=np.int8)
        self.target_ride_id = np.empty(0, dtype=np.int32)
        self.target_node_idx = np.empty(0, dtype=np.int32)
        self.walk_target_ride = np.empty(0, dtype=np.int32)
        self.preference_order = np.empty((0, config.NUM_RIDES), dtype=np.int16)
        self.balk_sec = np.empty((0, config.NUM_RIDES), dtype=np.float32)
        self.preferences = np.empty((0, config.NUM_RIDES), dtype=np.float32)
        self.must_do_remaining = np.empty((0, config.NUM_RIDES), dtype=np.bool_)
        self.ride_history = np.empty((0, config.NUM_RIDES), dtype=np.int16)
        self.rides_completed = np.empty(0, dtype=np.int32)

    def clear(self) -> None:
        self.count = 0
        self.party_size = np.empty(0, dtype=np.int32)
        self.spawn_sec = np.empty(0, dtype=np.int32)
        self.leave_sec = np.empty(0, dtype=np.int32)
        self.location_node_idx = np.empty(0, dtype=np.int32)
        self.effective_speed = np.empty(0, dtype=np.float32)
        self.state = np.empty(0, dtype=np.int8)
        self.target_ride_id = np.empty(0, dtype=np.int32)
        self.target_node_idx = np.empty(0, dtype=np.int32)
        self.walk_target_ride = np.empty(0, dtype=np.int32)
        self.preference_order = np.empty((0, config.NUM_RIDES), dtype=np.int16)
        self.balk_sec = np.empty((0, config.NUM_RIDES), dtype=np.float32)
        self.preferences = np.empty((0, config.NUM_RIDES), dtype=np.float32)
        self.must_do_remaining = np.empty((0, config.NUM_RIDES), dtype=np.bool_)
        self.ride_history = np.empty((0, config.NUM_RIDES), dtype=np.int16)
        self.rides_completed = np.empty(0, dtype=np.int32)

    def spawn_day(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        total_guests = max(1000, int(rng.normal(config.TOTAL_GUESTS_MEAN, config.TOTAL_GUESTS_STD)))
        entrance_idx = int(self.graph.entrance_node_idx)

        sizes: list[int] = []
        spawns: list[int] = []
        leaves: list[int] = []
        speeds: list[float] = []
        pref_rows: list[np.ndarray] = []
        must_do_rows: list[np.ndarray] = []
        schedules: list[tuple[int, int]] = []

        party_id = 0
        guests_assigned = 0

        while guests_assigned < total_guests:
            size = max(1, int(round(rng.normal(config.PARTY_SIZE_MEAN, config.PARTY_SIZE_STD))))
            guests_assigned += size

            spawn_sec = int(round(rng.normal(config.SPAWN_MEAN_SEC, config.SPAWN_STD_SEC)))
            spawn_sec = max(0, min(config.DAY_SECONDS - config.MIN_DWELL_SEC, spawn_sec))

            dwell = int(round(rng.normal(config.DWELL_MEAN_SEC, config.DWELL_STD_SEC)))
            dwell = max(config.MIN_DWELL_SEC, dwell)
            leave_sec = min(config.DAY_SECONDS, spawn_sec + dwell)

            must_do_count = int(rng.integers(0, 5))
            must_do = np.zeros(config.NUM_RIDES, dtype=np.bool_)
            if must_do_count > 0:
                must_do[rng.choice(config.NUM_RIDES, size=must_do_count, replace=False)] = True

            prefs = rng.uniform(0.1, 1.0, size=config.NUM_RIDES).astype(np.float32)
            prefs[must_do] *= config.MUST_DO_PREF_BOOST
            total = prefs.sum()
            if total > 0:
                prefs /= total

            sizes.append(size)
            spawns.append(spawn_sec)
            leaves.append(leave_sec)
            speeds.append(_party_effective_speed(rng, size))
            pref_rows.append(prefs)
            must_do_rows.append(must_do)
            schedules.append((spawn_sec, party_id))
            party_id += 1

        n = party_id
        self.count = n
        self.party_size = np.array(sizes, dtype=np.int32)
        self.spawn_sec = np.array(spawns, dtype=np.int32)
        self.leave_sec = np.array(leaves, dtype=np.int32)
        self.location_node_idx = np.full(n, entrance_idx, dtype=np.int32)
        self.effective_speed = np.array(speeds, dtype=np.float32)
        self.state = np.full(n, int(PartyState.WALKING), dtype=np.int8)
        self.target_ride_id = np.full(n, ROUTE_IDLE_CODE, dtype=np.int32)
        self.target_node_idx = np.full(n, entrance_idx, dtype=np.int32)
        self.walk_target_ride = np.full(n, -1, dtype=np.int32)
        self.preferences = np.stack(pref_rows)
        self.must_do_remaining = np.stack(must_do_rows)
        self.ride_history = np.zeros((n, config.NUM_RIDES), dtype=np.int16)
        self.rides_completed = np.zeros(n, dtype=np.int32)

        self.preference_order = np.empty((n, config.NUM_RIDES), dtype=np.int16)
        self.balk_sec = np.empty((n, config.NUM_RIDES), dtype=np.float32)
        for i in range(n):
            self.preference_order[i] = _compute_preference_order_row(
                self.preferences[i], self.must_do_remaining[i]
            )
            self.balk_sec[i] = _compute_balk_sec_array(self.preferences[i])

        return schedules

    def get(self, party_id: int) -> Party:
        """Materialize a Party view (for tests and debugging only)."""
        must_do = self.must_do_remaining[party_id].tolist()
        return Party(
            party_id=party_id,
            party_size=int(self.party_size[party_id]),
            spawn_sec=int(self.spawn_sec[party_id]),
            leave_sec=int(self.leave_sec[party_id]),
            effective_speed=float(self.effective_speed[party_id]),
            preferences=self.preferences[party_id].tolist(),
            must_do=must_do,
            must_do_remaining=list(must_do),
            ride_history=self.ride_history[party_id].tolist(),
            preference_order=self.preference_order[party_id].tolist(),
            balk_sec=self.balk_sec[party_id].tolist(),
            location_node=self.graph.idx_to_node(int(self.location_node_idx[party_id])),
            state=PartyState(int(self.state[party_id])),
            target_ride_id=int(self.target_ride_id[party_id]),
            target_node=self.graph.idx_to_node(int(self.target_node_idx[party_id])),
            rides_completed=int(self.rides_completed[party_id]),
        )

    def on_ride_completed(self, party_id: int, ride_id: int) -> None:
        self.ride_history[party_id, ride_id] += 1
        self.rides_completed[party_id] += 1
        if self.must_do_remaining[party_id, ride_id]:
            self.must_do_remaining[party_id, ride_id] = False
            self.preference_order[party_id] = _compute_preference_order_row(
                self.preferences[party_id], self.must_do_remaining[party_id]
            )
            self.balk_sec[party_id] = _compute_balk_sec_array(self.preferences[party_id])

    def should_leave(self, party_id: int, now_sec: int) -> bool:
        return now_sec >= self.leave_sec[party_id]

    def time_remaining(self, party_id: int, now_sec: int) -> int:
        return max(0, int(self.leave_sec[party_id]) - now_sec)
