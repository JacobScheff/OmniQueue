"""Advance both policies against one immutable historical environment."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from Park import config
from Park.comparison.history import HistoricalDay, TICK


@dataclass
class Party:
    now: int
    location: int = config.NUM_RIDES  # entrance / park exit
    completed: set[int] = field(default_factory=set)


@dataclass
class Choice:
    ride_id: int | None = None
    depart: int | None = None
    join: int | None = None
    detail: dict = field(default_factory=dict)


def run_policy(day: HistoricalDay, walks, policy, profile: dict, progress=None) -> dict:
    """Only execution sees future outcomes; policies receive a current board.

    A visit consumes the posted wait at queue entry, then the configured duration.
    Archive unavailability interrupts queues/rides without completion credit.
    Both parties must walk out by the recorded park close. No repeat visits.
    """
    started = time.perf_counter()
    state = Party(day.opening)
    events, decisions = [], []
    speed = profile["speed"]

    def walk_seconds(source, target):
        distance = walks.distances[source][target]
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"No verified walking path from {source} to {target}")
        return math.ceil(distance / speed)

    def event(kind, end, **extra):
        if end < state.now:
            raise ValueError("Replay time moved backwards")
        events.append(dict(kind=kind, start=state.now, end=end, location=state.location, **extra))
        state.now = end

    def idle(note):
        deadline = day.closing - walk_seconds(state.location, config.NUM_RIDES)
        event("idle", min((state.now // TICK + 1) * TICK, deadline), note=note)

    while state.now < day.closing:
        exit_walk = walk_seconds(state.location, config.NUM_RIDES)
        if state.now + exit_walk >= day.closing:
            break
        # Copy state so policy code cannot mutate the execution state.
        choice = policy.choose(Party(state.now, state.location, set(state.completed)),
                               day.board_at(state.now, observed=True))
        decisions.append(dict(now=state.now, location=state.location,
                              completed=sorted(state.completed), ride_id=choice.ride_id,
                              history_through=day.history_through(state.now), **choice.detail))
        if progress:
            progress(policy.name, state.now, len(state.completed), choice.detail)
        rid = choice.ride_id
        if rid is None:
            idle(choice.detail.get("reason", "No next ride; observe the next wait board"))
            continue
        if not 0 <= rid < config.NUM_RIDES or rid in state.completed:
            raise ValueError(f"{policy.name} chose an invalid/completed ride: {rid}")
        depart = state.now if choice.depart is None else max(state.now, int(choice.depart))
        travel = walk_seconds(state.location, rid)
        return_walk = walk_seconds(rid, config.NUM_RIDES)
        duration = int(config.RIDES[rid]["duration_sec"])
        if depart + travel + duration + return_walk > day.closing:
            idle("Insufficient time to reach this ride and walk out")
            continue
        if depart > state.now:
            event("idle", depart, note="Planned departure")
        source = state.location
        event("walk", state.now + travel, source=source, target=rid,
              meters=walks.distances[source][rid], ride_id=rid)
        state.location = rid
        join = max(state.now, int(choice.join or state.now))
        deadline = day.closing - return_walk
        if join >= deadline:
            event("idle", deadline, note="Queue entry would exceed departure deadline")
            break
        if join > state.now:
            event("idle", join, note="Planned queue entry", ride_id=rid)
        wait = day.board_at(state.now)[rid]
        if wait is None:
            events.append(dict(kind="unavailable", start=state.now, end=state.now,
                               location=rid, ride_id=rid))
            idle("Ride unavailable on arrival; no completion credited")
            continue
        board = state.now + wait
        finish = board + duration
        interrupted = day.interruption(rid, state.now, finish)
        end = min(finish, deadline, interrupted if interrupted is not None else finish)
        if min(board, end) > state.now:
            event("queue", min(board, end), ride_id=rid, posted_wait_sec=wait)
        if end > state.now:
            event("ride", end, ride_id=rid)
        if finish <= deadline and (interrupted is None or finish <= interrupted):
            state.completed.add(rid)
            events.append(dict(kind="complete", start=state.now, end=state.now,
                               location=rid, ride_id=rid))
        else:
            events.append(dict(kind="interrupted", start=state.now, end=state.now,
                               location=rid, ride_id=rid,
                               note="Unavailable archive bucket" if end == interrupted else "Departure deadline"))
    exit_walk = walk_seconds(state.location, config.NUM_RIDES)
    if exit_walk:
        source = state.location
        event("walk", state.now + exit_walk, source=source, target=config.NUM_RIDES,
              meters=walks.distances[source][config.NUM_RIDES])
    state.location = config.NUM_RIDES
    exit_time = state.now
    if state.now < day.closing:
        event("exited", day.closing)
    prefs = profile["preferences"]
    mass = sum(p*p for p in prefs)
    totals = {kind: sum(e["end"]-e["start"] for e in events if e["kind"] == kind)
              for kind in ("walk", "queue", "ride", "idle")}
    return dict(name=policy.name, events=events, decisions=decisions,
                metrics=dict(rides_completed=len(state.completed),
                             preference_score=sum(prefs[j] for j in state.completed),
                             preference_utility=100*sum(prefs[j]**2 for j in state.completed)/mass,
                             must_dos_completed=len(state.completed & set(profile["must_dos"])),
                             must_dos_assigned=len(profile["must_dos"]),
                             top3_hits=len(state.completed & set(sorted(range(len(prefs)), key=prefs.__getitem__, reverse=True)[:3])),
                             walk_m=sum(e.get("meters", 0) for e in events),
                             exit_time=exit_time, wall_time_sec=round(time.perf_counter()-started, 3),
                             **{kind+"_sec": value for kind, value in totals.items()}))
