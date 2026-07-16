"""Tests for opening-rush spawn and soft park close."""

from __future__ import annotations

import pytest

from simulator import native_backend_name, record_day

pytestmark = pytest.mark.skipif(
    native_backend_name() != "native", reason="C++ extension not built"
)


def _sample_at(samples, sec: int):
    for s in samples:
        if int(s.sec) == sec:
            return s
    raise AssertionError(f"no ride sample at sec={sec}")


def test_opening_rush_builds_waits_quickly():
    """Rope-drop rush should create meaningful waits within the first half hour."""
    rec = record_day(seed=42, sample_interval_sec=300)
    early = _sample_at(rec.ride_samples, 1800)  # +30 min
    waits = list(early.wait)
    mean_wait = sum(waits) / len(waits)
    max_wait = max(waits)
    queue_sum = sum(early.queue_len)
    assert mean_wait >= 5 * 60, f"expected mean wait >= 5 min at +30m, got {mean_wait / 60:.1f}"
    assert max_wait >= 15 * 60, f"expected some ride >= 15 min at +30m, got {max_wait / 60:.1f}"
    assert queue_sum >= 50, f"expected substantial queues at +30m, got {queue_sum}"


def test_park_close_still_has_long_waits():
    """Official close should not be empty — some rides keep long waits."""
    import _park_sim

    rec = record_day(seed=42, sample_interval_sec=300)
    close = _sample_at(rec.ride_samples, _park_sim.DAY_SECONDS)
    waits = list(close.wait)
    long_rides = sum(1 for w in waits if w >= 20 * 60)
    max_wait = max(waits)
    queue_sum = sum(close.queue_len)
    assert queue_sum > 0, "expected guests still queued at park close"
    assert long_rides >= 1, f"expected >=1 ride with wait >= 20 min at close, got {long_rides}"
    assert max_wait >= 25 * 60, f"expected max wait >= 25 min at close, got {max_wait / 60:.1f}"


def test_soft_close_router_exits_after_day_end():
    """After official close the heuristic always exits instead of picking another ride."""
    import _park_sim

    n = _park_sim.NUM_RIDES
    order = list(range(n))
    prefs = [1.0 / n] * n
    chosen = _park_sim.route_one_for_test(
        now_sec=_park_sim.DAY_SECONDS,
        leave_sec=_park_sim.DAY_SECONDS,
        node_idx=0,
        speed=1.4,
        preference_order=order,
        preferences=prefs,
        balk_sec=[40 * 60.0] * n,
        ride_history=[0] * n,
        open_mask=[1] * n,
        wait_times=[5 * 60.0] * n,
        durations=[120] * n,
        rand_u01=1.0,
    )
    assert chosen == -1  # kExitRideId
