"""Tests for ride breakdown and evacuation."""

import random

from park_types import Event, EventType
from rides import RideManager
from timing_wheel import TimingWheel


def test_breakdown_evacuation_order():
    rng = random.Random(0)
    rides = RideManager(rng)
    wheel = TimingWheel()

    ride = rides.get(0)
    ride.pending_board[10] = 100
    ride.pending_board[11] = 110
    ride.on_ride = [20, 21]

    route_now = rides.trigger_breakdown(wheel, 0, now_sec=500)
    assert 10 in route_now and 11 in route_now
    assert ride.status.name == "BROKEN"
    assert rides.has_evacuation_pending(0)

    first = rides.pop_evacuation(0)
    assert first in (10, 11)
    second = rides.pop_evacuation(0)
    assert second in (10, 11)
    assert first != second

    third = rides.pop_evacuation(0)
    assert third == 20
    fourth = rides.pop_evacuation(0)
    assert fourth == 21
    assert rides.pop_evacuation(0) is None
