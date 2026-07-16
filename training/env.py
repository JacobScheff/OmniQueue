"""Gymnasium wrapper around the C++ ParkEnv."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import _park_sim
from training.features import FLAT_OBS_DIM, NUM_ACTIONS


class ParkRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self._env = _park_sim.ParkEnv(seed)
        self._seed = seed
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FLAT_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        else:
            self._seed += 1
        obs = self._env.reset(self._seed)
        return np.asarray(obs.flat(), dtype=np.float32), {}

    def step(self, action: int):
        result = self._env.step(int(action))
        obs = np.zeros(FLAT_OBS_DIM, dtype=np.float32)
        if result.has_obs:
            obs = np.asarray(result.obs.flat(), dtype=np.float32)
        terminated = bool(result.done)
        truncated = False
        info = {}
        if terminated:
            info["metrics"] = {
                "rides_completed": result.metrics.rides_completed,
                "avg_wait_variance": result.metrics.avg_wait_variance(),
                "rides_per_party": result.metrics.rides_per_party(),
                "must_do_completion_rate": result.metrics.must_do_completion_rate(),
                "avg_preference_score_per_guest": result.metrics.avg_preference_score_per_guest(),
                "avg_must_do_latency_sec": result.metrics.avg_must_do_latency_sec(),
            }
        return obs, float(result.reward), terminated, truncated, info
