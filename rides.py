"""Ride state: boarding queues, breakdowns, and evacuation."""

from __future__ import annotations

import math
import random

import numpy as np

import config
from park_graph import get_park_graph
from park_types import Event, EventType, Ride, RideStatus
from timing_wheel import TimingWheel


class RideManager:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        graph = get_park_graph()
        self.rides: list[Ride] = []
        for cfg in config.get_ride_configs():
            self.rides.append(
                Ride(
                    ride_id=cfg["ride_id"],
                    name=cfg["name"],
                    capacity_per_sec=cfg["capacity_per_sec"],
                    duration_sec=cfg["duration_sec"],
                    breakdown_prob_sec=cfg["breakdown_prob_sec"],
                    node_id=cfg["node_id"],
                )
            )
        self.open_mask = np.ones(config.NUM_RIDES, dtype=bool)
        self.wait_arr = np.zeros(config.NUM_RIDES, dtype=np.float32)
        self.duration_arr = np.array([r.duration_sec for r in self.rides], dtype=np.int32)
        self.ride_node_idx = graph.ride_node_idx.copy()

    def refresh_router_cache(self) -> None:
        for i, ride in enumerate(self.rides):
            self.open_mask[i] = ride.status == RideStatus.OPEN
            self.wait_arr[i] = ride.current_wait_sec

    def get(self, ride_id: int) -> Ride:
        return self.rides[ride_id]

    def wait_times(self) -> list[float]:
        return self.wait_arr.tolist()

    def update_wait_estimates(self, now_sec: float) -> None:
        for i, ride in enumerate(self.rides):
            if ride.status == RideStatus.BROKEN:
                ride.current_wait_sec = 9999.0
            elif ride.capacity_per_sec <= 0:
                ride.current_wait_sec = 9999.0
            else:
                pending = len(ride.pending_board)
                on_ride_count = len(ride.on_ride)
                ahead = pending + on_ride_count
                until_board = max(0.0, ride.next_board_sec - now_sec)
                ride.current_wait_sec = until_board + ahead / ride.capacity_per_sec
            self.open_mask[i] = ride.status == RideStatus.OPEN
            self.wait_arr[i] = ride.current_wait_sec

    def schedule_boarding(self, wheel: TimingWheel, ride_id: int, party_id: int, now_sec: int) -> None:
        ride = self.rides[ride_id]
        if ride.status != RideStatus.OPEN:
            return
        start_sec = max(now_sec, int(math.ceil(ride.next_board_sec)))
        if start_sec <= now_sec:
            start_sec = now_sec + 1
        ride.next_board_sec = start_sec + 1.0 / ride.capacity_per_sec
        ride.pending_board[party_id] = start_sec
        wheel.schedule(
            start_sec,
            Event(EventType.RIDE_START, party_id=party_id, ride_id=ride_id, ride_generation=ride.generation),
        )

    def on_ride_start(self, ride_id: int, party_id: int, generation: int) -> bool:
        ride = self.rides[ride_id]
        if generation != ride.generation:
            return False
        if party_id not in ride.pending_board:
            return False
        if ride.status != RideStatus.OPEN:
            return False
        del ride.pending_board[party_id]
        ride.on_ride.append(party_id)
        return True

    def on_ride_complete(self, ride_id: int, party_id: int) -> bool:
        ride = self.rides[ride_id]
        if party_id in ride.on_ride:
            ride.on_ride.remove(party_id)
            return True
        return False

    def increment_incoming(self, ride_id: int) -> None:
        self.rides[ride_id].incoming_walkers += 1

    def decrement_incoming(self, ride_id: int) -> None:
        self.rides[ride_id].incoming_walkers = max(0, self.rides[ride_id].incoming_walkers - 1)

    def maybe_breakdown(self, wheel: TimingWheel, ride_id: int, now_sec: int) -> list[int] | None:
        ride = self.rides[ride_id]
        if ride.status == RideStatus.BROKEN:
            return None
        if self.rng.random() >= ride.breakdown_prob_sec:
            return None
        return self.trigger_breakdown(wheel, ride_id, now_sec)

    def trigger_breakdown(self, wheel: TimingWheel, ride_id: int, now_sec: int) -> list[int]:
        ride = self.rides[ride_id]
        if ride.status == RideStatus.BROKEN:
            return []

        ride.status = RideStatus.BROKEN
        repair = self.rng.randint(config.BREAKDOWN_REPAIR_MIN_SEC, config.BREAKDOWN_REPAIR_MAX_SEC)
        ride.broken_until_sec = now_sec + repair
        ride.cumulative_downtime_sec += repair
        ride.last_breakdown_sec = now_sec
        ride.generation += 1
        self.open_mask[ride_id] = False
        self.wait_arr[ride_id] = 9999.0

        wheel.schedule(
            ride.broken_until_sec,
            Event(EventType.BREAKDOWN_END, ride_id=ride_id, ride_generation=ride.generation),
        )

        route_now: list[int] = []
        for pid in list(ride.pending_board.keys()):
            route_now.append(pid)
            ride.evacuation.append(pid)
        ride.pending_board.clear()

        for pid in list(ride.on_ride):
            if pid not in ride.evacuating_on_ride:
                ride.evacuating_on_ride.append(pid)

        if not ride.evacuation_active and (ride.evacuation or ride.evacuating_on_ride):
            self._start_evacuation(wheel, ride_id, now_sec)

        return route_now

    def _start_evacuation(self, wheel: TimingWheel, ride_id: int, now_sec: int) -> None:
        ride = self.rides[ride_id]
        ride.evacuation_active = True
        wheel.schedule(
            now_sec + config.EVAC_INTERVAL_SEC,
            Event(EventType.EVACUATE_PARTY, ride_id=ride_id, ride_generation=ride.generation),
        )

    def on_breakdown_end(self, ride_id: int, generation: int, now_sec: int) -> bool:
        ride = self.rides[ride_id]
        if generation != ride.generation:
            return False
        if ride.status != RideStatus.BROKEN:
            return False
        ride.status = RideStatus.OPEN
        ride.next_board_sec = float(now_sec)
        self.open_mask[ride_id] = True
        return True

    def pop_evacuation(self, ride_id: int) -> int | None:
        ride = self.rides[ride_id]

        if ride.evacuation:
            return ride.evacuation.pop(0)

        if ride.evacuating_on_ride:
            pid = ride.evacuating_on_ride.pop(0)
            if pid in ride.on_ride:
                ride.on_ride.remove(pid)
            return pid

        return None

    def has_evacuation_pending(self, ride_id: int) -> bool:
        ride = self.rides[ride_id]
        return bool(ride.evacuation or ride.evacuating_on_ride)

    def schedule_next_evacuation(self, wheel: TimingWheel, ride_id: int, now_sec: int) -> None:
        ride = self.rides[ride_id]
        if self.has_evacuation_pending(ride_id):
            wheel.schedule(
                now_sec + config.EVAC_INTERVAL_SEC,
                Event(EventType.EVACUATE_PARTY, ride_id=ride_id, ride_generation=ride.generation),
            )
        else:
            ride.evacuation_active = False
