"""KPI types returned from simulation rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DayMetrics:
    total_parties: int = 0
    total_guests: int = 0
    rides_completed: int = 0
    parties_exited: int = 0
    breakdown_count: int = 0
    wait_variance_samples: list[float] = field(default_factory=list)
    mean_wait_samples: list[float] = field(default_factory=list)
    wall_time_sec: float = 0.0

    @property
    def rides_per_party(self) -> float:
        return self.rides_completed / max(1, self.total_parties)

    @property
    def rides_per_guest(self) -> float:
        if self.total_guests == 0:
            return 0.0
        return self.rides_completed / self.total_guests

    @property
    def avg_wait_variance(self) -> float:
        if not self.wait_variance_samples:
            return 0.0
        return sum(self.wait_variance_samples) / len(self.wait_variance_samples)
