"""KPI collection during simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import config


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


class MetricsCollector:
    def __init__(self) -> None:
        self.metrics = DayMetrics()
        self._next_sample_sec = 0

    def on_day_start(self, total_parties: int, total_guests: int) -> None:
        self.metrics.total_parties = total_parties
        self.metrics.total_guests = total_guests
        self._next_sample_sec = 0

    def maybe_sample(self, now_sec: int, wait_times: list[float]) -> None:
        if now_sec < self._next_sample_sec:
            return
        if now_sec > config.DAY_SECONDS:
            return

        valid = [w for w in wait_times if w < 9000]
        if valid:
            mean = sum(valid) / len(valid)
            var = sum((w - mean) ** 2 for w in valid) / len(valid)
            self.metrics.mean_wait_samples.append(mean)
            self.metrics.wait_variance_samples.append(var)

        self._next_sample_sec = now_sec + config.METRICS_SAMPLE_INTERVAL_SEC

    def on_ride_complete(self) -> None:
        self.metrics.rides_completed += 1

    def on_breakdown(self) -> None:
        self.metrics.breakdown_count += 1

    def on_party_exit(self) -> None:
        self.metrics.parties_exited += 1

    def finalize(self, wall_time: float) -> DayMetrics:
        self.metrics.wall_time_sec = wall_time
        return self.metrics
