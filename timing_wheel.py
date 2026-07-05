"""Bucket-array timing wheel for O(1) event scheduling."""

from __future__ import annotations

import config
from park_types import Event


class TimingWheel:
    """Schedule events at integer seconds using per-second buckets."""

    def __init__(self, day_seconds: int = config.DAY_SECONDS) -> None:
        self._day_seconds = day_seconds
        self._buckets: list[list[Event]] = [[] for _ in range(day_seconds + 1)]
        self._cursor = 0
        self._max_scheduled = -1
        self._current_sec = 0

    @property
    def current_sec(self) -> int:
        return self._current_sec

    def schedule(self, at_second: int, event: Event) -> None:
        if at_second < self._current_sec:
            at_second = self._current_sec
        if at_second > self._day_seconds:
            at_second = self._day_seconds
        self._buckets[at_second].append(event)
        if at_second > self._max_scheduled:
            self._max_scheduled = at_second

    def empty(self) -> bool:
        if self._max_scheduled < 0:
            return True
        if self._cursor > self._max_scheduled:
            return True
        while self._cursor <= self._max_scheduled and not self._buckets[self._cursor]:
            self._cursor += 1
        return self._cursor > self._max_scheduled

    def peek_time(self) -> int | None:
        if self.empty():
            return None
        cursor = self._cursor
        while cursor <= self._max_scheduled and not self._buckets[cursor]:
            cursor += 1
        return cursor if cursor <= self._max_scheduled else None

    def pop_next(self) -> tuple[int, list[Event]]:
        if self.empty():
            return self._current_sec, []

        while self._cursor <= self._max_scheduled and not self._buckets[self._cursor]:
            self._cursor += 1

        if self._cursor > self._max_scheduled:
            return self._current_sec, []

        self._current_sec = self._cursor
        events = self._buckets[self._cursor]
        self._buckets[self._cursor] = []
        sec = self._cursor
        self._cursor += 1
        return sec, events
