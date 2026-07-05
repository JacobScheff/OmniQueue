"""Tests for optional C++ native extension."""

import pytest

import config
from simulator import native_backend_name, run_day


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_native_run_day_smoke():
    metrics = run_day(seed=123, backend="native")
    assert metrics.total_parties > 0
    assert metrics.rides_completed > 0
    assert metrics.wall_time_sec >= 0


def test_python_run_day_smoke():
    metrics = run_day(seed=123, backend="python")
    assert metrics.total_parties > 0
    assert metrics.rides_completed > 0


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_native_requires_build_for_forced_backend():
    metrics = run_day(seed=0, backend="native")
    assert metrics.rides_per_party > 0
