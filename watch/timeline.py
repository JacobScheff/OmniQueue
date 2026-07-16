"""Timeline helpers for watch mode (decision marks + playhead)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from watch.session import DecisionMark


@dataclass
class TimelineState:
    """Playhead over a growing recording frontier."""

    playhead_sec: float = 0.0
    frontier_sec: float = 0.0
    paused: bool = True
    marks_scope: str = "focal"  # "focal" | "all"
    selected_mark_idx: int | None = None
    decisions: list[DecisionMark] = field(default_factory=list)

    def at_frontier(self, eps: float = 0.5) -> bool:
        return self.playhead_sec >= self.frontier_sec - eps

    def can_edit_prefs(self) -> bool:
        return self.paused and self.at_frontier()

    def visible_marks(self) -> list[tuple[int, DecisionMark]]:
        out: list[tuple[int, DecisionMark]] = []
        for i, m in enumerate(self.decisions):
            if self.marks_scope == "focal" and m.scope != "focal":
                continue
            out.append((i, m))
        return out

    def marks_at_or_before(self, sec: float) -> list[tuple[int, DecisionMark]]:
        return [(i, m) for i, m in self.visible_marks() if float(m.sec) <= sec + 1e-6]

    def nearest_mark(self, sec: float, max_dist: float = 90.0) -> int | None:
        best_i = None
        best_d = max_dist
        for i, m in self.visible_marks():
            d = abs(float(m.sec) - sec)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

def completion_counts_at(completions: list, party_id: int, sec: float, num_rides: int) -> list[int]:
    """Count focal ride completions with event.sec <= sec."""
    counts = [0] * num_rides
    for ev in completions:
        if int(ev.party_id) != party_id:
            continue
        if float(ev.sec) > sec:
            continue
        rid = int(ev.ride_id)
        if 0 <= rid < num_rides:
            counts[rid] += 1
    return counts


def scrub_to_frac(frac: float, day_seconds: float, frontier_sec: float) -> float:
    """Map a 0–1 slider fraction to a playhead capped at the recorded frontier."""
    frac = max(0.0, min(1.0, frac))
    return min(frontier_sec, frac * day_seconds)


def mark_index_for_click(
    marks: list[tuple[int, DecisionMark]],
    mx: int,
    slider_rect_x: int,
    slider_rect_w: int,
    day_seconds: float,
    hit_px: int = 10,
) -> int | None:
    if slider_rect_w <= 0 or day_seconds <= 0:
        return None
    best = None
    best_d = float(hit_px)
    for idx, m in marks:
        x = slider_rect_x + int(round((float(m.sec) / day_seconds) * slider_rect_w))
        d = abs(mx - x)
        if d <= best_d:
            best_d = d
            best = idx
    return best


def next_mark_after(decisions: list[DecisionMark], sec: float, scope: str) -> DecisionMark | None:
    """Next decision strictly after sec for the given marks scope."""
    best = None
    for m in decisions:
        if scope == "focal" and m.scope != "focal":
            continue
        if float(m.sec) > sec + 1e-6:
            if best is None or m.sec < best.sec:
                best = m
    return best


def insert_sorted_by_sec(decisions: list[DecisionMark], mark: DecisionMark) -> None:
    secs = [d.sec for d in decisions]
    i = bisect.bisect_right(secs, mark.sec)
    decisions.insert(i, mark)
