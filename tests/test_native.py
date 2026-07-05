"""Tests for the C++ native extension."""

import pytest

from simulator import native_backend_name, run_day


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_native_run_day_smoke():
    metrics = run_day(seed=123, backend="native")
    assert metrics.total_parties > 0
    assert metrics.rides_completed > 0
    assert metrics.wall_time_sec >= 0


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_native_metrics_sanity():
    metrics = run_day(seed=0, backend="native")
    assert metrics.rides_per_party > 0


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_run_day_default_backend():
    metrics = run_day(seed=42)
    assert metrics.total_parties > 0
    assert metrics.rides_completed > 0


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_exchange_batch_matches_step():
    import _park_sim

    seed = 7
    action = 36  # idle wander

    env_step = _park_sim.ParkEnv(seed)
    env_step.reset(seed)
    step_rewards: list[float] = []
    while True:
        result = env_step.step(action)
        step_rewards.append(float(result.reward))
        if result.done:
            step_metrics = result.metrics
            break

    env_batch = _park_sim.ParkEnv(seed)
    env_batch.reset(seed)
    batch_rewards: list[float] = []
    pending: list[int] = []
    while True:
        batch = env_batch.exchange_batch(pending, 1)
        pending = []
        if batch.n_rewards > 0:
            batch_rewards.extend(float(x) for x in batch.rewards)
        if batch.episode_done:
            batch_metrics = batch.metrics
            break
        pending = [action]

    assert len(batch_rewards) == len(step_rewards)
    assert batch_rewards == pytest.approx(step_rewards, rel=1e-5, abs=1e-5)
    assert batch_metrics.rides_completed == step_metrics.rides_completed
    assert batch_metrics.avg_wait_variance() == pytest.approx(step_metrics.avg_wait_variance(), rel=1e-5)
