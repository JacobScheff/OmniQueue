"""Bounded time-expanded DP with a required-subset feasibility oracle.

All times are absolute integer seconds. Bounds are scheduling assumptions,
not calibrated probabilities of completing a whole route.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
import time


@dataclass(frozen=True)
class Ride:
    id: int
    name: str
    duration: int
    preference: float
    must_do: bool = False
    completed: bool = False
    opens: int = 0
    queue_closes: int = 2**62
    finish_by: int = 2**62


@dataclass
class Context:
    rides: list[Ride]
    now: int
    leave: int
    # Indices 0..n-1 are rides, n is the current location, n+1 the exit.
    distances: list[list[float]]
    wait_at: object  # Callable[[ride index, queue entry second], int | None]
    speed: float = 1.4
    walk_cost: float = 2.0  # utility points / km
    buffer: int = 300
    step: int = 300
    beam: int = 300
    seconds: float = 10.0
    score_gap: float = 3.0
    operating_at: object = None  # Optional Callable[[ride index, time], probability]
    risk_cost: float = 5.0  # utility points * -log(entry operating probability)


class SearchTimeout(Exception):
    pass


@dataclass(frozen=True)
class Label:
    mask: int
    last: int
    ready: int
    score: float
    stops: tuple


def plan(ctx: Context) -> dict:
    """Return up to three checked full itineraries, or an explicit failure status."""
    started = time.perf_counter()
    deadline = started + ctx.seconds
    n = len(ctx.rides)
    if (not n or ctx.leave <= ctx.now or not 0 < ctx.step <= 3600
            or ctx.speed <= 0 or ctx.beam < 1 or ctx.seconds <= 0
            or ctx.walk_cost < 0 or ctx.buffer < 0 or ctx.score_gap < 0
            or not math.isfinite(ctx.risk_cost) or ctx.risk_cost < 0):
        raise ValueError("Invalid time window, walking settings, or search settings")
    if len({r.id for r in ctx.rides}) != n:
        raise ValueError("Ride IDs must be unique")
    if len(ctx.distances) != n + 2 or any(len(row) != n + 2 for row in ctx.distances):
        raise ValueError("Distance matrix must include rides, current location, and exit")
    if any(not math.isfinite(r.preference) or r.preference < 0 or r.duration <= 0
           for r in ctx.rides):
        raise ValueError("Preferences must be finite/nonnegative; durations positive")
    required = sum(1 << j for j, r in enumerate(ctx.rides) if r.must_do and not r.completed)
    available = sum(1 << j for j, r in enumerate(ctx.rides) if not r.completed)
    mass = sum(r.preference**2 for r in ctx.rides)
    values = [100 * r.preference**2 / mass if mass else 100 / n for r in ctx.rides]
    ticks = (ctx.leave - ctx.now) // ctx.step + 1
    checks = 0

    def check_time():
        nonlocal checks
        checks += 1
        if checks % 256 == 0 and time.perf_counter() >= deadline:
            raise SearchTimeout

    def walking(i, j):
        meters = ctx.distances[i][j]
        return math.ceil(meters / ctx.speed) if math.isfinite(meters) and meters >= 0 else None

    def exit_info(i, t):
        duration = walking(i, n + 1)
        if duration is None or t + duration > ctx.leave:
            return None
        return {"exit_time": t + duration, "exit_walk_m": ctx.distances[i][n + 1],
                "exit_walk_sec": duration}

    @lru_cache(maxsize=150_000)
    def leg(i, j, t):
        check_time()
        r = ctx.rides[j]
        walk = walking(i, j)
        if walk is None:
            return None
        arrival = t + walk
        join = max(arrival, r.opens)
        # Queue entries use the decision grid, too. Waiting at the entrance
        # prevents timing-phase false negatives from optional intermediate stops.
        join = ctx.now + math.ceil((join - ctx.now) / ctx.step) * ctx.step
        if join >= min(r.queue_closes, ctx.leave):
            return None
        wait = ctx.wait_at(j, join)
        if wait is None:
            return None
        if not math.isfinite(wait) or wait < 0:
            raise ValueError("Waits must be finite/nonnegative or None")
        finish = join + math.ceil(wait) + r.duration + ctx.buffer
        ready = ctx.now + math.ceil((finish - ctx.now) / ctx.step) * ctx.step
        if finish > r.finish_by or exit_info(j, ready) is None:
            return None
        return (j, t, arrival, join, finish, ready, ctx.distances[i][j], walk, math.ceil(wait))

    @lru_cache(maxsize=None)
    def witness(mask, i, t):
        check_time()
        if not mask:
            return () if exit_info(i, t) else None
        if t >= ctx.leave:
            return None
        for j in sorted((j for j in range(n) if mask & (1 << j)),
                        key=lambda j: ctx.rides[j].queue_closes):
            stop = leg(i, j, t)
            if stop is not None:
                tail = witness(mask ^ (1 << j), j, stop[5])
                if tail is not None:
                    return (stop,) + tail
        return witness(mask, i, t + ctx.step)

    def extend(label, stop):
        j, _, _, _, finish, ready, meters, _, _ = stop
        utility = values[j] * math.exp(-0.15 * (finish - ctx.now) / 54000)
        probability = ctx.operating_at(j, stop[3]) if ctx.operating_at else 1.0
        if probability is None or not 0 < probability <= 1:
            raise ValueError('Schedulable rides need an operating probability in (0, 1]')
        risk_penalty = ctx.risk_cost * -math.log(probability)
        score = label.score + utility - ctx.walk_cost * meters / 1000 - risk_penalty
        return Label(label.mask | (1 << j), j, ready, score, label.stops + (stop,))

    def complete(label, tail):
        for stop in tail:
            # The witness may idle before its first scheduled departure.
            if stop[1] > label.ready:
                label = Label(label.mask, label.last, stop[1], label.score, label.stops)
            label = extend(label, stop)
        out = exit_info(label.last, label.ready)
        if out is None:
            return None
        score = label.score - ctx.walk_cost * out["exit_walk_m"] / 1000
        return (score, label, out)

    initial = Label(0, n, ctx.now, 0.0, ())
    routes = {}

    def retain(result):
        if result is None:
            return
        signature = tuple(s[0] for s in result[1].stops)
        if signature not in routes or result[0] > routes[signature][0]:
            routes[signature] = result
        if len(routes) > 300:
            for key in sorted(routes, key=lambda key: routes[key][0])[:100]:
                del routes[key]

    timed_out = False
    expanded = 0
    beam_pruned = False
    try:
        first = witness(required, n, ctx.now)
        if first is None:
            return {"status": "uncertified", "routes": [], "elapsed_sec": round(time.perf_counter()-started, 3),
                    "reason": "No required-only schedule fits the grid, bounds, coverage and departure. Try finer time steps or fewer constraints."}
        retain(complete(initial, first))
        buckets = [{} for _ in range(ticks + 1)]

        def add(label):
            nonlocal beam_pruned
            tick = (label.ready - ctx.now) // ctx.step
            if tick >= len(buckets):
                return
            bucket = buckets[tick]
            # Preserve different prefixes that converge to the same state.
            key = (label.mask, label.last, tuple(s[0] for s in label.stops[:2]))
            if key not in bucket or label.score > bucket[key].score:
                bucket[key] = label
            if len(bucket) > ctx.beam * 2:
                keep = heapq.nlargest(ctx.beam, bucket.items(), key=lambda p: p[1].score)
                bucket.clear()
                bucket.update(keep)
                beam_pruned = True

        add(initial)
        for tick in range(ticks):
            bucket = buckets[tick]
            labels = heapq.nlargest(ctx.beam, bucket.values(), key=lambda x: x.score)
            beam_pruned |= len(bucket) > ctx.beam
            buckets[tick] = {}
            for label in labels:
                check_time()
                for j in range(n):
                    if not available & (1 << j) or label.mask & (1 << j):
                        continue
                    expanded += 1
                    stop = leg(label.last, j, label.ready)
                    if stop is None:
                        continue
                    successor = extend(label, stop)
                    tail = witness(required & ~successor.mask, j, successor.ready)
                    if tail is not None:
                        retain(complete(successor, tail))
                        add(successor)
                # WAIT is essential for non-FIFO forecast wait curves.
                if label.ready + ctx.step <= ctx.leave:
                    add(Label(label.mask, label.last, label.ready + ctx.step,
                              label.score, label.stops))
    except SearchTimeout:
        timed_out = True

    # Validation must finish even when the search used its entire budget.
    deadline = math.inf
    ordered = sorted(routes.values(), key=lambda r: r[0], reverse=True)
    selected = []
    def different(a, b):
        a = [s[0] for s in a.stops]
        b = [s[0] for s in b.stops]
        edges_a, edges_b = set(zip([-1]+a, a)), set(zip([-1]+b, b))
        return a[:3] != b[:3] and 1-len(edges_a & edges_b)/max(1, len(edges_a | edges_b)) >= 0.3
    for item in ordered:
        if item[0] < ordered[0][0] - ctx.score_gap:
            break
        if all(different(item[1], old[1]) for old in selected):
            selected.append(item)
        if len(selected) == 3:
            break
    output = []
    for score, label, out in selected:
        # Validate the complete schedule again, including required coverage.
        assert label.mask & required == required
        stops = []
        previous, ready = n, ctx.now
        for s in label.stops:
            assert s[1] >= ready and leg(previous, s[0], s[1]) == s
            j, depart, arrival, join, finish, ready, meters, walk_sec, wait_sec = s
            stops.append(dict(ride_id=ctx.rides[j].id, name=ctx.rides[j].name,
                              must_do=ctx.rides[j].must_do, depart=depart, arrival=arrival,
                              join=join, finish=finish, ready=ready, walk_m=meters,
                              walk_sec=walk_sec, wait_sec=wait_sec,
                              operating_probability=ctx.operating_at(j, join) if ctx.operating_at else 1.0))
            previous = j
        output.append(dict(score=round(score, 3), stops=stops,
                           total_walk_m=sum(s[6] for s in label.stops)+out["exit_walk_m"],
                           **out))
    return dict(status="feasible_under_assumptions" if output else "search_incomplete",
                routes=output, elapsed_sec=round(time.perf_counter()-started, 3),
                expanded=expanded, timed_out=timed_out, beam_pruned=beam_pruned,
                search_complete=not timed_out and not beam_pruned,
                required_count=required.bit_count(), step_sec=ctx.step,
                assumptions="Wait estimates plus buffer; forecast availability and park-wide hours. Ride-specific hours, eligibility and interruptions are unverified. No calibrated success probability.")
