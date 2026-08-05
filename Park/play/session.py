"""In-memory session store for interactive play runs (not persisted)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from Park.play.scoring import FocalScore, ParkScore


@dataclass
class FocalProfile:
    spawn_sec: int = 0
    leave_sec: int = 14 * 3600
    preference_weights: np.ndarray = field(
        default_factory=lambda: np.ones(34, dtype=np.float32)
    )
    must_dos: np.ndarray = field(default_factory=lambda: np.zeros(34, dtype=np.uint8))
    distance_preference: float = 0.5

    def copy(self) -> "FocalProfile":
        return FocalProfile(
            spawn_sec=int(self.spawn_sec),
            leave_sec=int(self.leave_sec),
            preference_weights=np.array(self.preference_weights, dtype=np.float32, copy=True),
            must_dos=np.array(self.must_dos, dtype=np.uint8, copy=True),
            distance_preference=float(self.distance_preference),
        )


@dataclass
class RunSettings:
    seed: int
    kind: str  # human | ai_compare | benchmark
    crowd_router: str  # heuristic | ppo
    focal_router: str  # human | heuristic | ppo
    checkpoint: str | None = None
    label: str = ""

    def summary(self) -> str:
        base = f"seed={self.seed} crowd={self.crowd_router} focal={self.focal_router}"
        if self.checkpoint:
            base += f" ckpt={self.checkpoint}"
        if self.label:
            base += f" [{self.label}]"
        return base


@dataclass
class SessionRun:
    settings: RunSettings
    profile: FocalProfile
    park: ParkScore
    focal: FocalScore
    itinerary: list[tuple[int, int]] = field(default_factory=list)  # (sec, ride_id)
    recording: Any | None = None  # optional DayRecording for replay segments


class SessionStore:
    def __init__(self) -> None:
        self.runs: list[SessionRun] = []

    def add(self, run: SessionRun) -> None:
        self.runs.append(run)

    def clear(self) -> None:
        self.runs.clear()

    def by_kind(self, kind: str) -> list[SessionRun]:
        return [r for r in self.runs if r.settings.kind == kind]
