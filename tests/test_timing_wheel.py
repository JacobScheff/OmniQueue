"""Tests for the min-heap timing wheel."""

from park_types import Event, EventType
from timing_wheel import TimingWheel


def test_schedule_and_pop_order():
    wheel = TimingWheel()
    wheel.schedule(10, Event(EventType.PARTY_SPAWN, party_id=1))
    wheel.schedule(5, Event(EventType.PARTY_SPAWN, party_id=2))
    wheel.schedule(5, Event(EventType.PARTY_SPAWN, party_id=3))

    t1, e1 = wheel.pop_next()
    assert t1 == 5
    assert len(e1) == 2

    t2, e2 = wheel.pop_next()
    assert t2 == 10
    assert len(e2) == 1


def test_same_second_fifo():
    wheel = TimingWheel()
    for i in range(5):
        wheel.schedule(100, Event(EventType.PARTY_SPAWN, party_id=i))

    t, events = wheel.pop_next()
    assert t == 100
    assert [e.party_id for e in events] == [0, 1, 2, 3, 4]
