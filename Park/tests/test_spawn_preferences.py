"""Spawn preferences should track real ride popularity with slight noise."""

import numpy as np
import pytest

import Park.config as config
from Park.simulator import native_backend_name


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_mean_preferences_follow_ride_popularity():
    """Across many parties, Space Mountain should outrank Explorer Canoes on average."""
    import _park_sim

    names = [r["name"] for r in config.RIDES]
    space_idx = names.index("Space Mountain")
    canoe_idx = names.index("Davy Crockett's Explorer Canoes")
    assert config.RIDES[space_idx]["popularity"] > config.RIDES[canoe_idx]["popularity"]

    sums = np.zeros(config.NUM_RIDES, dtype=np.float64)
    seen: set[tuple[float, ...]] = set()
    # A few full days of routing obs; dedupe by rounded preference vector.
    for seed in (0, 1, 2, 7, 11):
        env = _park_sim.ParkEnv(seed)
        obs = env.reset(seed)
        for _ in range(400):
            prefs = np.asarray(obs.guest, dtype=np.float64)[: config.NUM_RIDES]
            assert abs(prefs.sum() - 1.0) < 1e-3
            key = tuple(np.round(prefs, 6).tolist())
            if key not in seen:
                seen.add(key)
                sums += prefs
            result = env.step(35)  # idle
            if result.done:
                break
            obs = result.obs

    assert len(seen) >= 200
    means = sums / len(seen)
    assert means[space_idx] > means[canoe_idx] * 2.0

    pops = np.array([float(r["popularity"]) for r in config.RIDES], dtype=np.float64)
    pop_ranks = pops.argsort().argsort().astype(np.float64)
    mean_ranks = means.argsort().argsort().astype(np.float64)
    corr = np.corrcoef(pop_ranks, mean_ranks)[0, 1]
    assert corr > 0.7


def test_config_popularity_table_complete():
    assert len(config.RIDES) == config.NUM_RIDES
    for ride in config.RIDES:
        assert "popularity" in ride
        assert ride["popularity"] > 0
    assert 0.0 < config.PREF_POPULARITY_NOISE < 1.0
