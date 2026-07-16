"""In-memory session store for watch runs (not persisted)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from play.scoring import FocalScore, ParkScore
from play.session import FocalProfile


@dataclass
class DecisionMark:
    """One PPO routing decision captured for timeline inspection."""

    sec: int
    scope: str  # "focal" | "crowd"
    party_id: int
    action: int
    probs: np.ndarray  # (NUM_ACTIONS,)


@dataclass
class WatchSettings:
    seed: int
    crowd_router: str  # heuristic | ppo
    checkpoint: str
    label: str = ""

    def summary(self) -> str:
        base = f"seed={self.seed} crowd={self.crowd_router} focal=ppo"
        if self.checkpoint:
            base += f" ckpt={self.checkpoint}"
        if self.label:
            base += f" [{self.label}]"
        return base


@dataclass
class WatchRun:
    settings: WatchSettings
    profile: FocalProfile
    park: ParkScore
    focal: FocalScore
    decisions: list[DecisionMark] = field(default_factory=list)
    recording: Any | None = None


@dataclass
class WatchStore:
    runs: list[WatchRun] = field(default_factory=list)

    def add(self, run: WatchRun) -> None:
        self.runs.append(run)
