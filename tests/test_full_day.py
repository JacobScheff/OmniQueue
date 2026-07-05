"""Full-day integration smoke test (small guest count)."""

import config
from simulator import run_day


def test_full_day_smoke(monkeypatch):
    monkeypatch.setattr(config, "TOTAL_GUESTS_MEAN", 500)
    monkeypatch.setattr(config, "TOTAL_GUESTS_STD", 50)
    metrics = run_day(seed=123)
    assert metrics.total_parties > 0
    assert metrics.rides_completed > 0
    assert metrics.wall_time_sec > 0
