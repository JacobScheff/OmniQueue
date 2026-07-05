"""Shared types for the discrete event simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, IntFlag


class EventType(IntEnum):
    PARTY_SPAWN = 0
    ARRIVE_AT_DESTINATION = 1
    RIDE_START = 2
    RIDE_COMPLETE = 3
    BREAKDOWN_END = 4
    EVACUATE_PARTY = 5


class PartyState(IntFlag):
    WALKING = 1
    IN_QUEUE = 2
    ON_RIDE = 4
    EVACUATING = 8
    EXITED = 16


class RideStatus(IntEnum):
    OPEN = 0
    BROKEN = 1


# Sentinel ride id meaning "exit the park"
EXIT_RIDE_ID = -1
# Router output: idle wander (resolved to random node in simulator)
ROUTE_IDLE_CODE = -2


@dataclass(slots=True)
class Event:
    type: EventType
    party_id: int = -1
    ride_id: int = -1
    # Used to invalidate stale events after breakdown generation bumps.
    ride_generation: int = 0
    # Secondary payload (e.g. target node for arrivals).
    payload: int = 0


@dataclass
class Party:
    party_id: int
    party_size: int
    spawn_sec: int
    leave_sec: int
    effective_speed: float
    preferences: list[float]
    must_do: list[bool]
    must_do_remaining: list[bool]
    ride_history: list[int]
    preference_order: list[int]
    balk_sec: list[float]
    location_node: int
    state: PartyState = PartyState.WALKING
    target_ride_id: int = EXIT_RIDE_ID
    target_node: int = 0
    arrival_sec: int = 0
    rides_completed: int = 0
    rng_state: tuple = field(default_factory=tuple)


@dataclass
class Ride:
    ride_id: int
    name: str
    capacity_per_sec: float
    duration_sec: int
    breakdown_prob_sec: float
    node_id: int
    status: RideStatus = RideStatus.OPEN
    broken_until_sec: int = 0
    generation: int = 0
    next_board_sec: float = 0.0
    current_wait_sec: float = 0.0
    incoming_walkers: int = 0
    cumulative_downtime_sec: int = 0
    pending_board: dict[int, int] = field(default_factory=dict)
    on_ride: list[int] = field(default_factory=list)
    evacuation: list[int] = field(default_factory=list)
    evacuating_on_ride: list[int] = field(default_factory=list)
    evacuation_active: bool = False
    last_breakdown_sec: int = -1
