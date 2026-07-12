"""Tests for the C++ native extension."""

import pytest

from simulator import native_backend_name, run_day


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_native_run_day_smoke():
    metrics = run_day(seed=123, backend="native")
    assert metrics.total_parties > 0
    assert metrics.rides_completed > 0
    assert metrics.wall_time_sec > 0


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


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_dense_wait_var_reward_is_not_constant_step_penalty():
    """Every routing step should use live wait variance, not a flat -0.001."""
    import _park_sim

    seed = 7
    env = _park_sim.ParkEnv(seed)
    env.reset(seed)

    rewards: list[float] = []
    # Send parties to rides so queues (and wait variance) actually form.
    action = 0
    while True:
        result = env.step(action)
        rewards.append(float(result.reward))
        if result.done:
            break
        action = (action + 1) % 34

    mid = rewards[:-1]
    assert len(mid) > 1000
    # Dense wait-var penalties vary with park state; not a constant step tax.
    assert len({round(r, 6) for r in mid}) > 1
    # Mid-day congestion should produce some clearly negative wait-var penalties.
    assert min(mid) < -0.0005


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_preference_reward_flushed_after_ride_complete():
    """Must-do / preference flush can push a post-completion step reward positive."""
    import _park_sim

    seed = 11
    env = _park_sim.ParkEnv(seed)
    env.reset(seed)

    rewards: list[float] = []
    rides_seen = 0
    action = 0
    while True:
        result = env.step(action)
        rewards.append(float(result.reward))
        if result.done:
            rides_seen = int(result.metrics.rides_completed)
            break
        action = (action + 1) % 34

    assert rides_seen > 0
    # Must-do bonus (0.005) exceeds typical dense wait-var step penalties mid-day.
    assert any(r > 0.0 for r in rewards[:-1]), (
        "expected at least one positive mid-episode reward from preference/must-do flush "
        f"(got max={max(rewards[:-1]):.6f})"
    )
