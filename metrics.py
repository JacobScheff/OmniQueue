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
    must_dos_assigned: int = 0
    must_dos_completed: int = 0
    preference_score_sum: float = 0.0
    must_do_latency_sum_sec: float = 0.0
    must_do_latency_count: int = 0

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

    @property
    def avg_mean_wait(self) -> float:
        if not self.mean_wait_samples:
            return 0.0
        return sum(self.mean_wait_samples) / len(self.mean_wait_samples)

    @property
    def must_do_completion_rate(self) -> float:
        if self.must_dos_assigned <= 0:
            return 1.0
        return self.must_dos_completed / self.must_dos_assigned

    @property
    def avg_preference_score_per_guest(self) -> float:
        if self.total_guests <= 0:
            return 0.0
        return self.preference_score_sum / self.total_guests

    @property
    def avg_must_do_latency_sec(self) -> float:
        if self.must_do_latency_count <= 0:
            return 0.0
        return self.must_do_latency_sum_sec / self.must_do_latency_count
