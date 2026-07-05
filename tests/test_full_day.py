"""Full-day integration smoke test."""

import pytest

from simulator import native_backend_name, run_day


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_full_day_smoke():
    metrics = run_day(seed=123)
    assert metrics.total_parties > 0
    assert metrics.rides_completed > 0
    assert metrics.wall_time_sec > 0
