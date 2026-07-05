"""Party pool: spawning, preferences, must-do lists, and speed."""

from __future__ import annotations

import math

import numpy as np

import config
from park_graph import ParkGraph, get_park_graph
from park_types import Party, PartyState


def _compute_balk_sec(preferences: list[float]) -> list[float]:
    return [
        config.BASE_BALK_SEC + config.BALK_SCALE * (p ** config.BALK_PREF_EXP)
        for p in preferences
    ]


def _compute_preference_order(preferences: list[float], must_do_remaining: list[bool]) -> list[int]:
    order = list(range(config.NUM_RIDES))

    def sort_key(rid: int) -> tuple[int, float, int]:
        must = 0 if must_do_remaining[rid] else 1
        return (must, -preferences[rid], rid)

    order.sort(key=sort_key)
    return order


def _party_effective_speed(rng: np.random.Generator, party_size: int) -> float:
    member_speeds = rng.lognormal(config.MEMBER_SPEED_LOG_MU, config.MEMBER_SPEED_LOG_SIGMA, size=party_size)
    return float(np.min(member_speeds))


def _generate_preferences(rng: np.random.Generator, must_do: list[bool]) -> list[float]:
    prefs = rng.uniform(0.1, 1.0, size=config.NUM_RIDES).tolist()
    for i in range(config.NUM_RIDES):
        if must_do[i]:
            prefs[i] *= config.MUST_DO_PREF_BOOST
    total = sum(prefs)
    if total > 0:
        prefs = [p / total for p in prefs]
    return prefs


class PartyPool:
    def __init__(self, graph: ParkGraph | None = None) -> None:
        self.graph = graph or get_park_graph()
        self.parties: list[Party] = []

    def clear(self) -> None:
        self.parties.clear()

    def spawn_day(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Create parties and return list of (spawn_sec, party_id) schedules."""
        total_guests = max(1000, int(rng.normal(config.TOTAL_GUESTS_MEAN, config.TOTAL_GUESTS_STD)))
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
            must_do_indices = rng.choice(config.NUM_RIDES, size=must_do_count, replace=False) if must_do_count > 0 else []
            must_do = [False] * config.NUM_RIDES
            for idx in must_do_indices:
                must_do[int(idx)] = True

            preferences = _generate_preferences(rng, must_do)
            must_do_remaining = list(must_do)
            ride_history = [0] * config.NUM_RIDES
            preference_order = _compute_preference_order(preferences, must_do_remaining)
            balk_sec = _compute_balk_sec(preferences)

            party = Party(
                party_id=party_id,
                party_size=size,
                spawn_sec=spawn_sec,
                leave_sec=leave_sec,
                effective_speed=_party_effective_speed(rng, size),
                preferences=preferences,
                must_do=must_do,
                must_do_remaining=must_do_remaining,
                ride_history=ride_history,
                preference_order=preference_order,
                balk_sec=balk_sec,
                location_node=self.graph.entrance_node,
                state=PartyState.WALKING,
            )
            self.parties.append(party)
            schedules.append((spawn_sec, party_id))
            party_id += 1

        return schedules

    def get(self, party_id: int) -> Party:
        return self.parties[party_id]

    def active_count(self) -> int:
        return sum(1 for p in self.parties if p.state != PartyState.EXITED)

    def on_ride_completed(self, party: Party, ride_id: int) -> None:
        party.ride_history[ride_id] += 1
        party.rides_completed += 1
        if party.must_do_remaining[ride_id]:
            party.must_do_remaining[ride_id] = False
            party.preference_order = _compute_preference_order(party.preferences, party.must_do_remaining)
            party.balk_sec = _compute_balk_sec(party.preferences)

    def time_remaining(self, party: Party, now_sec: int) -> int:
        return max(0, party.leave_sec - now_sec)

    def should_leave(self, party: Party, now_sec: int) -> bool:
        return now_sec >= party.leave_sec

    def refresh_preference_order(self, party: Party) -> None:
        party.preference_order = _compute_preference_order(party.preferences, party.must_do_remaining)
