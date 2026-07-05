"""Event handlers and main DES simulation loop."""

from __future__ import annotations

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
from park_types import EXIT_RIDE_ID, Event, EventType, PartyState, RideStatus


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
        self.walking_to_ride: dict[int, set[int]] = defaultdict(set)
        self._walk_targets: dict[int, tuple[int | None, int]] = {}

    def run_day(self) -> DayMetrics:
        t0 = time.perf_counter()
        self._reset()

        schedules = self.parties.spawn_day(self.np_rng)
        total_guests = sum(self.parties.get(pid).party_size for _, pid in schedules)
        self.metrics.on_day_start(len(schedules), total_guests)

        for spawn_sec, party_id in schedules:
            self.wheel.schedule(spawn_sec, Event(EventType.PARTY_SPAWN, party_id=party_id))

        while not self.wheel.empty():
            now_sec, events = self.wheel.pop_next()
            if now_sec > config.DAY_SECONDS:
                break

            self.rides.update_wait_estimates(now_sec)
            self.rides.invalidate_router_cache()
            self.metrics.maybe_sample(now_sec, self.rides.wait_times())

            for ride_id in range(config.NUM_RIDES):
                route_now = self.rides.maybe_breakdown(self.wheel, ride_id, now_sec)
                if route_now is not None:
                    self.metrics.on_breakdown()
                    self._on_breakdown(ride_id, route_now, now_sec)

            deciding: list[int] = []

            for event in events:
                match event.type:
                    case EventType.PARTY_SPAWN:
                        party = self.parties.get(event.party_id)
                        party.location_node = self.graph.entrance_node
                        deciding.append(event.party_id)
                    case EventType.ARRIVE_AT_DESTINATION:
                        deciding.extend(self._handle_arrive(event.party_id, now_sec))
                    case EventType.RIDE_START:
                        deciding.extend(self._handle_ride_start(event, now_sec))
                    case EventType.RIDE_COMPLETE:
                        deciding.extend(
                            self._handle_ride_complete(event.party_id, event.ride_id, now_sec)
                        )
                    case EventType.BREAKDOWN_END:
                        self.rides.on_breakdown_end(event.ride_id, event.ride_generation, now_sec)
                    case EventType.EVACUATE_PARTY:
                        deciding.extend(self._handle_evacuate(event, now_sec))

            if deciding:
                self._route_parties(deciding, now_sec)

        return self.metrics.finalize(time.perf_counter() - t0)

    def _reset(self) -> None:
        self.parties.clear()
        self.rides = RideManager(self.rng)
        self.metrics = MetricsCollector()
        self.wheel = TimingWheel()
        self.walking_to_ride = defaultdict(set)
        self._walk_targets = {}

    def _on_breakdown(self, ride_id: int, route_at_entrance: list[int], now_sec: int) -> None:
        entrance = self.rides.get(ride_id).node_id

        for pid in list(self.walking_to_ride.get(ride_id, set())):
            self._cancel_walk(pid)
            self._route_parties([pid], now_sec)

        for pid in route_at_entrance:
            party = self.parties.get(pid)
            party.location_node = entrance
            party.state = PartyState.EVACUATING

        if route_at_entrance:
            self._route_parties(route_at_entrance, now_sec)

    def _cancel_walk(self, party_id: int) -> None:
        if party_id not in self._walk_targets:
            return
        target_ride, _ = self._walk_targets.pop(party_id)
        if target_ride is not None and target_ride >= 0:
            self.walking_to_ride[target_ride].discard(party_id)
            self.rides.decrement_incoming(target_ride)

    def _handle_arrive(self, party_id: int, now_sec: int) -> list[int]:
        party = self.parties.get(party_id)
        if party_id in self._walk_targets:
            target_ride, _ = self._walk_targets.pop(party_id)
            if target_ride is not None and target_ride >= 0:
                self.walking_to_ride[target_ride].discard(party_id)
                self.rides.decrement_incoming(target_ride)

        party.location_node = party.target_node

        if party.target_ride_id == EXIT_RIDE_ID:
            party.state = PartyState.EXITED
            self.metrics.on_party_exit()
            return []

        if party.target_ride_id is None:
            return [party_id]

        ride_id = party.target_ride_id
        ride = self.rides.get(ride_id)

        if ride.status == RideStatus.BROKEN:
            return [party_id]

        party.state = PartyState.IN_QUEUE
        self.rides.schedule_boarding(self.wheel, ride_id, party_id, now_sec)
        return []

    def _handle_ride_start(self, event: Event, now_sec: int) -> list[int]:
        if not self.rides.on_ride_start(event.ride_id, event.party_id, event.ride_generation):
            return []

        party = self.parties.get(event.party_id)
        party.state = PartyState.ON_RIDE
        ride = self.rides.get(event.ride_id)
        self.wheel.schedule(
            now_sec + ride.duration_sec,
            Event(EventType.RIDE_COMPLETE, party_id=event.party_id, ride_id=event.ride_id),
        )
        return []

    def _handle_ride_complete(self, party_id: int, ride_id: int, now_sec: int) -> list[int]:
        party = self.parties.get(party_id)
        self.rides.on_ride_complete(ride_id, party_id)
        self.parties.on_ride_completed(party, ride_id)
        self.metrics.on_ride_complete()
        party.location_node = self.rides.get(ride_id).node_id
        party.state = PartyState.WALKING
        return [party_id]

    def _handle_evacuate(self, event: Event, now_sec: int) -> list[int]:
        ride = self.rides.get(event.ride_id)
        if event.ride_generation != ride.generation:
            return []

        pid = self.rides.pop_evacuation(event.ride_id)
        if pid is None:
            ride.evacuation_active = False
            return []

        party = self.parties.get(pid)
        party.location_node = ride.node_id
        party.state = PartyState.WALKING

        if self.rides.has_evacuation_pending(event.ride_id):
            self.rides.schedule_next_evacuation(self.wheel, event.ride_id, now_sec)

        return [pid]

    def _route_parties(self, party_ids: list[int], now_sec: int) -> None:
        unique_ids: list[int] = []
        seen: set[int] = set()
        for pid in party_ids:
            if pid in seen:
                continue
            party = self.parties.get(pid)
            if party.state == PartyState.EXITED:
                continue
            unique_ids.append(pid)
            seen.add(pid)

        if not unique_ids:
            return

        decisions = self.router.route_batch(
            unique_ids, self.parties, self.rides, self.graph, now_sec, self.np_rng
        )

        for party_id, target in decisions:
            self._assign_route(party_id, target, now_sec)

    def _assign_route(self, party_id: int, target, now_sec: int) -> None:
        party = self.parties.get(party_id)
        if party.state == PartyState.EXITED:
            return

        self._cancel_walk(party_id)

        if target == EXIT_RIDE_ID:
            party.target_ride_id = EXIT_RIDE_ID
            dest = self.graph.entrance_node
            party.target_node = dest
            walk = self.graph.party_walk_time(party.location_node, dest, party.effective_speed)
            party.state = PartyState.WALKING
            party.arrival_sec = now_sec + walk
            self._walk_targets[party_id] = (EXIT_RIDE_ID, dest)
            self.wheel.schedule(
                party.arrival_sec,
                Event(EventType.ARRIVE_AT_DESTINATION, party_id=party_id),
            )
            return

        if target is None:
            dest = self.graph.random_idle_node(self.np_rng, party.location_node)
            party.target_ride_id = None
            party.target_node = dest
            walk = self.graph.party_walk_time(party.location_node, dest, party.effective_speed)
            party.state = PartyState.WALKING
            party.arrival_sec = now_sec + walk
            self._walk_targets[party_id] = (None, dest)
            self.wheel.schedule(
                party.arrival_sec,
                Event(EventType.ARRIVE_AT_DESTINATION, party_id=party_id),
            )
            return

        ride_id = int(target)
        ride = self.rides.get(ride_id)
        dest = ride.node_id
        party.target_ride_id = ride_id
        party.target_node = dest
        walk = self.graph.party_walk_time(party.location_node, dest, party.effective_speed)
        party.state = PartyState.WALKING
        party.arrival_sec = now_sec + walk
        self._walk_targets[party_id] = (ride_id, dest)
        self.walking_to_ride[ride_id].add(party_id)
        self.rides.increment_incoming(ride_id)
        self.wheel.schedule(
            party.arrival_sec,
            Event(EventType.ARRIVE_AT_DESTINATION, party_id=party_id),
        )


def run_day(seed: int = 0, router: str | None = None) -> DayMetrics:
    r = get_router(router) if router else None
    sim = Simulator(seed=seed, router=r)
    return sim.run_day()
