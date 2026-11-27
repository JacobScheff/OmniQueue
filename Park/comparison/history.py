"""Recorded wait boards, kept separate from the policies' visible history."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import gzip
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from Park import config
from Park.companion.server.ride_map import build_config_name_index, resolve_ride_index

ZONE = ZoneInfo("America/Los_Angeles")
TICK = 300
# Version-1 archive vocabulary, defined by Wait Time Predictions/ml/data.py.
OVERFLOW = 61


@dataclass
class HistoricalDay:
    key: str
    timestamps: list[int]
    waits: list[list[int | None]]
    closing: int
    interpolated: dict[int, int]
    source: str = ""

    @property
    def opening(self):
        return self.timestamps[0]

    def index_at(self, moment: int) -> int:
        index = bisect_right(self.timestamps, moment) - 1
        if index < 0 or moment >= self.closing or moment - self.timestamps[index] >= TICK:
            return -1
        return index

    def board_at(self, moment: int, *, observed: bool = False) -> list[int | None]:
        index = self.index_at(moment)
        if index < 0:
            return [None] * config.NUM_RIDES
        tick = self.timestamps[index]
        # An interpolated row is only knowable after its right endpoint arrives.
        if observed and self.interpolated.get(tick, tick) > moment:
            return [None] * config.NUM_RIDES
        return list(self.waits[index])

    def history_through(self, moment: int) -> int | None:
        index = bisect_right(self.timestamps, moment) - 1
        while index >= 0:
            tick = self.timestamps[index]
            if self.interpolated.get(tick, tick) <= moment:
                return tick
            index -= 1
        return None

    def interruption(self, ride_id: int, start: int, end: int) -> int | None:
        """First unavailable archive bucket while queued or on a ride."""
        for index in range(bisect_right(self.timestamps, start), len(self.timestamps)):
            tick = self.timestamps[index]
            if tick >= end:
                break
            if self.waits[index][ride_id] is None:
                return tick
        return None


def load_day(path: Path, requested: str | None = None, *, today: date | None = None) -> HistoricalDay:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        dataset = json.load(source)
    if dataset.get("version") != 1:
        raise ValueError("Unsupported wait archive version")
    days = dataset.get("days", {})
    today = today or datetime.now(ZONE).date()
    if requested:
        date.fromisoformat(requested)
        key = requested
    else:
        yesterday = (today - timedelta(days=1)).isoformat()
        candidates = [key for key in days if key <= yesterday]
        if not candidates:
            raise ValueError("No completed historical days. Export the wait dataset first.")
        key = yesterday if yesterday in days else max(candidates)
    if key not in days:
        raise ValueError(f"No complete cached day for {key}. Export it with Wait Time Predictions/ml/data.py.")
    day = days[key]
    ticks = day["timestamps"]
    expected = day.get("expected_timestamps", ticks)
    if (not ticks or ticks != sorted(set(ticks)) or ticks != expected
            or len(day["tokens"]) != len(ticks)):
        raise ValueError(f"{key} does not contain its full scheduled day")
    names = build_config_name_index()
    slots = {}
    for column, ride in enumerate(day["rides"]):
        rid = resolve_ride_index(ride.get("id"), ride.get("name", ""), names)
        if rid is not None:
            if rid in slots:
                raise ValueError(f"Ambiguous archive mapping for {config.RIDES[rid]['name']}")
            slots[rid] = column
    if not slots:
        raise ValueError("The archive contains no mapped Park rides")
    waits = []
    for tokens in day["tokens"]:
        if len(tokens) != len(day["rides"]) or any(t < 0 or t > 65 for t in tokens):
            raise ValueError(f"{key} contains missing or invalid attraction observations")
        waits.append([tokens[slots[j]] * 5 * 60 if j in slots and tokens[slots[j]] < OVERFLOW
                      else None for j in range(config.NUM_RIDES)])
    closing = int(day.get("closing_at", ticks[-1] + TICK))
    if closing <= ticks[-1] or closing > ticks[-1] + TICK:
        raise ValueError("Closing time does not match the archive's last scheduled bucket")
    return HistoricalDay(key, ticks, waits, closing,
                         {int(t): int(pair[1]) for t, pair in day.get("interpolated_ticks", {}).items()},
                         str(path.resolve()))
