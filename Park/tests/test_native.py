"""Tests for the C++ native extension."""

import pytest

from Park.simulator import native_backend_name, run_day


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
    assert metrics.must_dos_assigned > 0
    assert 0.0 <= metrics.must_do_completion_rate <= 1.0
    assert metrics.avg_preference_score_per_guest > 0.0
    assert metrics.must_do_latency_count > 0
    assert metrics.avg_must_do_latency_sec > 0.0


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_run_day_default_backend():
    metrics = run_day(seed=42)
    assert metrics.total_parties > 0
    assert metrics.rides_completed > 0


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_exchange_batch_matches_step():
    import _park_sim

    seed = 7
    action = 35  # idle wander (NUM_ACTIONS-1; rides 0..33, exit 34, idle 35)

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
    assert batch_metrics.must_do_completion_rate() == pytest.approx(
        step_metrics.must_do_completion_rate(), rel=1e-5
    )


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_pref_reward_has_no_wait_variance_term():
    """Step rewards should not track park wait variance (preference objective)."""
    import _park_sim
    from Park.training.features import FLAT_OBS_DIM, GUEST_FEAT_DIM, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM

    seed = 7
    env = _park_sim.ParkEnv(seed)
    obs0 = env.reset(seed)
    flat = list(obs0.flat())
    assert len(flat) == FLAT_OBS_DIM
    env_offset = GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    # Wait-variance slot withheld from the policy.
    assert flat[env_offset + 2] == pytest.approx(0.0)

    rewards: list[float] = []
    action = 0
    while True:
        result = env.step(action)
        rewards.append(float(result.reward))
        if result.done:
            break
        action = (action + 1) % 34

    mid = rewards[:-1]
    assert len(mid) > 1000
    # Urgency alone is a small non-positive tax; completions can push steps positive.
    assert min(mid) > -0.01
    assert any(r > 0.0 for r in mid)


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
    assert any(r > 0.0 for r in rewards[:-1]), (
        "expected at least one positive mid-episode reward from preference/must-do flush "
        f"(got max={max(rewards[:-1]):.6f})"
    )


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_obs_includes_remaining_pref_and_elapsed():
    import _park_sim
    from Park.training.features import (
        GUEST_FEAT_ELAPSED_SINCE_SPAWN,
        GUEST_FEAT_REMAINING_PREF_MASS,
        GUEST_FEAT_DIM,
    )

    env = _park_sim.ParkEnv(0)
    obs = env.reset(0)
    guest = list(obs.guest)
    assert len(guest) == GUEST_FEAT_DIM
    assert 0.0 <= guest[GUEST_FEAT_REMAINING_PREF_MASS] <= 1.0 + 1e-5
    assert guest[GUEST_FEAT_ELAPSED_SINCE_SPAWN] >= 0.0
