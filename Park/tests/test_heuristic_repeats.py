"""Tests for heuristic ride-repeat dampening."""

from __future__ import annotations

from collections import defaultdict

import pytest

from simulator import native_backend_name, record_day

pytestmark = pytest.mark.skipif(
    native_backend_name() != "native", reason="C++ extension not built"
)


def _base_route_args(n: int):
    import _park_sim

    assert n == _park_sim.NUM_RIDES
    order = list(range(n))
    prefs = [1.0 / n] * n
    # Make the first few slightly preferred so Pass 2 top-K is meaningful.
    prefs[0] = 0.08
    prefs[1] = 0.06
    prefs[2] = 0.05
    rest = (1.0 - sum(prefs[:3])) / (n - 3)
    for i in range(3, n):
        prefs[i] = rest
    balk = [40 * 60.0] * n
    history = [0] * n
    open_mask = [1] * n
    waits = [10 * 60.0] * n
    durations = [120] * n
    return {
        "now_sec": 0,
        "leave_sec": 54_000,
        "node_idx": 0,  # entrance — not a ride node
        "speed": 1.4,
        "preference_order": order,
        "preferences": prefs,
        "balk_sec": balk,
        "ride_history": history,
        "open_mask": open_mask,
        "wait_times": waits,
        "durations": durations,
        "rand_u01": 1.0,  # skip idle in Pass 4
    }


def test_route_prefers_fresh_over_heavy_repeat():
    import _park_sim

    n = _park_sim.NUM_RIDES
    args = _base_route_args(n)
    args["ride_history"][0] = 5
    args["ride_history"][1] = 0
    # Both under balk; Pass 1 must pick the fresh ride (1), not top pref (0).
    chosen = _park_sim.route_one_for_test(**args)
    assert chosen == 1


def test_route_prefers_fresh_over_single_top_repeat():
    import _park_sim

    n = _park_sim.NUM_RIDES
    args = _base_route_args(n)
    args["ride_history"][0] = 1
    args["ride_history"][1] = 0
    chosen = _park_sim.route_one_for_test(**args)
    assert chosen == 1


def test_route_short_wait_allows_repeat_when_others_long():
    import _park_sim

    n = _park_sim.NUM_RIDES
    args = _base_route_args(n)
    # Every ride already done once; all waits over balk except one short ride.
    args["ride_history"] = [1] * n
    args["wait_times"] = [50 * 60.0] * n
    short_ride = 7
    args["wait_times"][short_ride] = 5 * 60.0
    # Exhaust Pass 2 budgets so Pass 3 is the path under test.
    args["ride_history"] = [3] * n
    chosen = _park_sim.route_one_for_test(**args)
    assert chosen == short_ride


def test_recorded_day_diversifies_rides_per_party():
    """Parties with several completions should not concentrate on 1–2 rides."""
    rec = record_day(seed=42, sample_interval_sec=300)
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for ev in rec.ride_completions:
        counts[int(ev.party_id)][int(ev.ride_id)] += 1

    active = {
        pid: ride_counts
        for pid, ride_counts in counts.items()
        if sum(ride_counts.values()) >= 6
    }
    assert len(active) > 50

    unique_ratios = []
    max_shares = []
    for ride_counts in active.values():
        total = sum(ride_counts.values())
        unique_ratios.append(len(ride_counts) / total)
        max_shares.append(max(ride_counts.values()) / total)

    mean_unique_ratio = sum(unique_ratios) / len(unique_ratios)
    mean_max_share = sum(max_shares) / len(max_shares)
    # Novelty bias: most completions should be distinct rides; no single ride dominates.
    assert mean_unique_ratio > 0.55
    assert mean_max_share < 0.35
