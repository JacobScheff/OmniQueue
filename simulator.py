"""Event handlers and main DES simulation loop."""

from __future__ import annotations

import os
import random
import time
from collections import defaultdict

import numpy as np

import config
from metrics import DayMetrics, MetricsCollector
from park_graph import ParkGraph, get_park_graph
from parties import PartyPool
from rides import RideManager
from router.base import Router, get_router
from timing_wheel import TimingWheel
from park_types import EXIT_RIDE_ID, Event, EventType, PartyState, RideStatus, ROUTE_IDLE_CODE


def _native_available() -> bool:
    try:
        import _park_sim  # type: ignore[import-not-found]

        return bool(_park_sim.is_available())
    except ImportError:
        return False


def _metrics_from_native(result) -> DayMetrics:
    return DayMetrics(
        total_parties=result.total_parties,
        total_guests=result.total_guests,
        rides_completed=result.rides_completed,
        parties_exited=result.parties_exited,
        breakdown_count=result.breakdown_count,
        wait_variance_samples=list(result.wait_variance_samples),
        mean_wait_samples=list(result.mean_wait_samples),
        wall_time_sec=result.wall_time_sec,
    )


def native_backend_name() -> str:
    return "native" if _native_available() else "unavailable"


class Simulator:
    def __init__(
        self,
        seed: int = 0,
        router: Router | None = None,
        graph: ParkGraph | None = None,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.graph = graph or get_park_graph()
        self.router = router or get_router()
        self.parties = PartyPool(self.graph)
        self.rides = RideManager(self.rng)
        self.metrics = MetricsCollector()
        self.wheel = TimingWheel()

    def run_day(self) -> DayMetrics:
        t0 = time.perf_counter()
        self._reset()

        schedules = self.parties.spawn_day(self.np_rng)
        total_guests = int(self.parties.party_size.sum())
        self.metrics.on_day_start(self.parties.count, total_guests)

        spawn_event = EventType.PARTY_SPAWN
        for spawn_sec, party_id in schedules:
            self.wheel.schedule(spawn_sec, Event(spawn_event, party_id=party_id))

        entrance_idx = int(self.graph.entrance_node_idx)

        while not self.wheel.empty():
            now_sec, events = self.wheel.pop_next()
            if now_sec > config.DAY_SECONDS:
                break

            self.rides.update_wait_estimates(now_sec)
            self.metrics.maybe_sample(now_sec, self.rides.wait_times())

            for ride_id in range(config.NUM_RIDES):
                route_now = self.rides.maybe_breakdown(self.wheel, ride_id, now_sec)
                if route_now is not None:
                    self.metrics.on_breakdown()
                    self._on_breakdown(ride_id, route_now, now_sec)

            deciding: list[int] = []

            for event in events:
                if event.type == EventType.PARTY_SPAWN:
                    self.parties.location_node_idx[event.party_id] = entrance_idx
                    deciding.append(event.party_id)
                elif event.type == EventType.ARRIVE_AT_DESTINATION:
                    deciding.extend(self._handle_arrive(event.party_id, now_sec))
                elif event.type == EventType.RIDE_START:
                    self._handle_ride_start(event, now_sec)
                elif event.type == EventType.RIDE_COMPLETE:
                    deciding.append(event.party_id)
                    self._handle_ride_complete(event.party_id, event.ride_id)
                elif event.type == EventType.BREAKDOWN_END:
                    self.rides.on_breakdown_end(event.ride_id, event.ride_generation, now_sec)
                elif event.type == EventType.EVACUATE_PARTY:
                    deciding.extend(self._handle_evacuate(event, now_sec))

            if deciding:
                self._route_parties(deciding, now_sec)

        return self.metrics.finalize(time.perf_counter() - t0)

    def _reset(self) -> None:
        self.parties.clear()
        self.rides = RideManager(self.rng)
        self.metrics = MetricsCollector()
        self.wheel = TimingWheel()

    def _on_breakdown(self, ride_id: int, route_at_entrance: list[int], now_sec: int) -> None:
        ride_node_idx = int(self.rides.ride_node_idx[ride_id])
        walker_pids = np.where(self.parties.walk_target_ride == ride_id)[0]
        for pid in walker_pids:
            self._cancel_walk(int(pid))
        if walker_pids.size:
            self._route_parties(walker_pids.tolist(), now_sec)

        if route_at_entrance:
            for pid in route_at_entrance:
                self.parties.location_node_idx[pid] = ride_node_idx
                self.parties.state[pid] = int(PartyState.EVACUATING)
            self._route_parties(route_at_entrance, now_sec)

    def _cancel_walk(self, party_id: int) -> None:
        target_ride = int(self.parties.walk_target_ride[party_id])
        if target_ride >= 0:
            self.rides.decrement_incoming(target_ride)
            self.parties.walk_target_ride[party_id] = -1

    def _handle_arrive(self, party_id: int, now_sec: int) -> list[int]:
        self._cancel_walk(party_id)
        self.parties.location_node_idx[party_id] = self.parties.target_node_idx[party_id]

        target_ride = int(self.parties.target_ride_id[party_id])
        if target_ride == EXIT_RIDE_ID:
            self.parties.state[party_id] = int(PartyState.EXITED)
            self.metrics.on_party_exit()
            return []

        if target_ride == ROUTE_IDLE_CODE:
            return [party_id]

        if self.rides.rides[target_ride].status == RideStatus.BROKEN:
            return [party_id]

        self.parties.state[party_id] = int(PartyState.IN_QUEUE)
        self.rides.schedule_boarding(self.wheel, target_ride, party_id, now_sec)
        return []

    def _handle_ride_start(self, event: Event, now_sec: int) -> None:
        if not self.rides.on_ride_start(event.ride_id, event.party_id, event.ride_generation):
            return
        self.parties.state[event.party_id] = int(PartyState.ON_RIDE)
        duration = int(self.rides.duration_arr[event.ride_id])
        self.wheel.schedule(
            now_sec + duration,
            Event(EventType.RIDE_COMPLETE, party_id=event.party_id, ride_id=event.ride_id),
        )

    def _handle_ride_complete(self, party_id: int, ride_id: int) -> None:
        self.rides.on_ride_complete(ride_id, party_id)
        self.parties.on_ride_completed(party_id, ride_id)
        self.metrics.on_ride_complete()
        self.parties.location_node_idx[party_id] = self.rides.ride_node_idx[ride_id]
        self.parties.state[party_id] = int(PartyState.WALKING)

    def _handle_evacuate(self, event: Event, now_sec: int) -> list[int]:
        ride = self.rides.get(event.ride_id)
        if event.ride_generation != ride.generation:
            return []

        pid = self.rides.pop_evacuation(event.ride_id)
        if pid is None:
            ride.evacuation_active = False
            return []

        self.parties.location_node_idx[pid] = self.rides.ride_node_idx[event.ride_id]
        self.parties.state[pid] = int(PartyState.WALKING)

        if self.rides.has_evacuation_pending(event.ride_id):
            self.rides.schedule_next_evacuation(self.wheel, event.ride_id, now_sec)

        return [pid]

    def _route_parties(self, party_ids: list[int], now_sec: int) -> None:
        if not party_ids:
            return

        ids = np.fromiter(party_ids, dtype=np.int32, count=len(party_ids))
        exited = int(PartyState.EXITED)
        active_mask = self.parties.state[ids] != exited
        if not np.any(active_mask):
            return
        unique_ids = np.unique(ids[active_mask]).tolist()

        decisions = self.router.route_batch(
            unique_ids, self.parties, self.rides, self.graph, now_sec, self.np_rng
        )

        for party_id, target in decisions:
            self._assign_route(party_id, target, now_sec)

    def _assign_route(self, party_id: int, target, now_sec: int) -> None:
        if self.parties.state[party_id] == int(PartyState.EXITED):
            return

        self._cancel_walk(party_id)
        from_idx = int(self.parties.location_node_idx[party_id])
        speed = float(self.parties.effective_speed[party_id])

        if target == EXIT_RIDE_ID:
            dest_idx = int(self.graph.entrance_node_idx)
            walk = self.graph.party_walk_sec(from_idx, dest_idx, speed)
            self.parties.target_ride_id[party_id] = EXIT_RIDE_ID
            self.parties.target_node_idx[party_id] = dest_idx
            self.parties.state[party_id] = int(PartyState.WALKING)
            self.wheel.schedule(
                now_sec + walk,
                Event(EventType.ARRIVE_AT_DESTINATION, party_id=party_id),
            )
            return

        if target is None:
            dest_idx = self.graph.random_idle_node_idx(self.np_rng, from_idx)
            walk = self.graph.party_walk_sec(from_idx, dest_idx, speed)
            self.parties.target_ride_id[party_id] = ROUTE_IDLE_CODE
            self.parties.target_node_idx[party_id] = dest_idx
            self.parties.state[party_id] = int(PartyState.WALKING)
            self.wheel.schedule(
                now_sec + walk,
                Event(EventType.ARRIVE_AT_DESTINATION, party_id=party_id),
            )
            return

        ride_id = int(target)
        dest_idx = int(self.rides.ride_node_idx[ride_id])
        walk = self.graph.party_walk_to_ride_sec(from_idx, ride_id, speed)
        self.parties.target_ride_id[party_id] = ride_id
        self.parties.target_node_idx[party_id] = dest_idx
        self.parties.state[party_id] = int(PartyState.WALKING)
        self.parties.walk_target_ride[party_id] = ride_id
        self.rides.increment_incoming(ride_id)
        self.wheel.schedule(
            now_sec + walk,
            Event(EventType.ARRIVE_AT_DESTINATION, party_id=party_id),
        )


def run_day(
    seed: int = 0,
    router: str | None = None,
    backend: str | None = None,
) -> DayMetrics:
    """Run one simulated park day.

    backend:
      - ``auto`` (default): use C++ extension when built, else Python
      - ``native``: require C++ extension
      - ``python``: force Python simulator
    """
    selected = backend or os.environ.get("OMNIQUEUE_BACKEND", "auto")
    router_name = router or config.ROUTER

    if selected in ("auto", "native"):
        if router_name not in ("heuristic", None):
            if selected == "native":
                raise ValueError("C++ backend currently supports heuristic routing only.")
        elif _native_available():
            import _park_sim  # type: ignore[import-not-found]

            return _metrics_from_native(_park_sim.run_day(seed))

        if selected == "native":
            raise ImportError(
                "C++ extension _park_sim is not built. Run: pip install -e ."
            )

    r = get_router(router_name) if router_name else None
    sim = Simulator(seed=seed, router=r)
    return sim.run_day()
