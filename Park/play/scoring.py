"""Preference / park KPI helpers for interactive play runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FocalScore:
    rides_completed: int = 0
    preference_score: float = 0.0
    must_dos_assigned: int = 0
    must_dos_completed: int = 0
    top3_hits: int = 0
    exit_sec: int = -1
    exited: bool = False

    @property
    def must_do_rate(self) -> float:
        if self.must_dos_assigned <= 0:
            return 1.0
        return self.must_dos_completed / self.must_dos_assigned

    @property
    def top3_rate(self) -> float:
        if self.rides_completed <= 0:
            return 0.0
        return self.top3_hits / self.rides_completed


@dataclass
class ParkScore:
    total_parties: int = 0
    total_guests: int = 0
    rides_completed: int = 0
    rides_per_party: float = 0.0
    mean_wait: float = 0.0
    wait_variance: float = 0.0
    breakdown_count: int = 0
    wall_time_sec: float = 0.0


def normalize_weights(weights: np.ndarray, must_dos: np.ndarray | None = None) -> np.ndarray:
    """L1-normalize preference weights; optional must-do boost matches spawn semantics."""
    import config

    w = np.asarray(weights, dtype=np.float64).copy()
    if w.shape != (config.NUM_RIDES,):
        raise ValueError(f"weights must have length {config.NUM_RIDES}")
    w = np.maximum(w, 0.0)
    if must_dos is not None:
        md = np.asarray(must_dos, dtype=np.uint8)
        w = w * np.where(md > 0, config.MUST_DO_PREF_BOOST, 1.0)
    s = float(w.sum())
    if s <= 0:
        w[:] = 1.0
        s = float(config.NUM_RIDES)
    return (w / s).astype(np.float32)


def focal_from_native(stats) -> FocalScore:
    return FocalScore(
        rides_completed=int(stats.rides_completed),
        preference_score=float(stats.preference_score),
        must_dos_assigned=int(stats.must_dos_assigned),
        must_dos_completed=int(stats.must_dos_completed),
        top3_hits=int(stats.top3_hits),
        exit_sec=int(stats.exit_sec),
        exited=bool(stats.exited),
    )


def park_from_native(metrics) -> ParkScore:
    mean_wait = 0.0
    wait_var = 0.0
    mean_samples = list(getattr(metrics, "mean_wait_samples", []) or [])
    var_samples = list(getattr(metrics, "wait_variance_samples", []) or [])
    if mean_samples:
        mean_wait = sum(mean_samples) / len(mean_samples)
    if hasattr(metrics, "avg_wait_variance"):
        wait_var = float(metrics.avg_wait_variance())
    elif var_samples:
        wait_var = sum(var_samples) / len(var_samples)
    rides = int(metrics.rides_completed)
    parties = int(metrics.total_parties)
    return ParkScore(
        total_parties=parties,
        total_guests=int(metrics.total_guests),
        rides_completed=rides,
        rides_per_party=rides / max(1, parties),
        mean_wait=mean_wait,
        wait_variance=wait_var,
        breakdown_count=int(metrics.breakdown_count),
        wall_time_sec=float(getattr(metrics, "wall_time_sec", 0.0) or 0.0),
    )


def format_focal_line(score: FocalScore) -> str:
    return (
        f"rides={score.rides_completed}  pref={score.preference_score:.3f}  "
        f"must-do={score.must_dos_completed}/{score.must_dos_assigned}  "
        f"top3={score.top3_hits}/{score.rides_completed}"
    )


def format_park_line(score: ParkScore) -> str:
    return (
        f"rides/party={score.rides_per_party:.2f}  "
        f"mean wait={score.mean_wait / 60.0:.1f}m  "
        f"wait var={score.wait_variance / 1e6:.3f}e6"
    )
