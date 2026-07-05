"""Min-heap timing wheel for the discrete event simulator."""

from __future__ import annotations

import heapq
from itertools import count

from park_types import Event


class TimingWheel:
    """Schedule events at integer seconds; pop earliest second batch."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Event]] = []
        self._counter = count()
        self._current_sec = 0

    @property
    def current_sec(self) -> int:
        return self._current_sec

    def schedule(self, at_second: int, event: Event) -> None:
        if at_second < self._current_sec:
            at_second = self._current_sec
        heapq.heappush(self._heap, (at_second, next(self._counter), event))

    def empty(self) -> bool:
        return not self._heap

    def peek_time(self) -> int | None:
        if not self._heap:
            return None
        return self._heap[0][0]

    def pop_next(self) -> tuple[int, list[Event]]:
        if not self._heap:
            return self._current_sec, []

        next_sec = self._heap[0][0]
        self._current_sec = next_sec
        events: list[Event] = []

        while self._heap and self._heap[0][0] == next_sec:
            _, _, event = heapq.heappop(self._heap)
            events.append(event)

        return next_sec, events
