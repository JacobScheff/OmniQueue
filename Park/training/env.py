"""Gymnasium wrapper around the C++ ParkEnv (personal planner: 1 focal + heuristic crowd)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import _park_sim
from Park.training.features import FLAT_OBS_DIM, NUM_ACTIONS


class ParkRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, num_focals: int = 1):
        super().__init__()
        self._env = _park_sim.ParkEnv(seed)
        self._seed = seed
        self._num_focals = int(num_focals)
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
        n_focals = self._num_focals
        if options and "num_focals" in options:
            n_focals = int(options["num_focals"])
        self._env.reset_personal(self._seed, n_focals)
        # First pending observation via empty exchange.
        result = self._env.exchange_batch([], 1)
        if result.n_obs <= 0:
            obs = np.zeros(FLAT_OBS_DIM, dtype=np.float32)
        else:
            obs = np.asarray(result.obs, dtype=np.float32).reshape(FLAT_OBS_DIM)
        return obs, {}

    def step(self, action: int):
        result = self._env.exchange_batch([int(action)], 1)
        obs = np.zeros(FLAT_OBS_DIM, dtype=np.float32)
        reward = 0.0
        if result.n_rewards > 0:
            reward = float(np.asarray(result.rewards, dtype=np.float32)[0])
        if result.n_obs > 0:
            obs = np.asarray(result.obs, dtype=np.float32).reshape(-1, FLAT_OBS_DIM)[0]
        terminated = bool(result.episode_done)
        truncated = False
        info = {}
        if terminated:
            personal = self._env.personal_stats()
            info["metrics"] = {
                "rides_completed": personal.rides_completed,
                "avg_wait_variance": (
                    result.metrics.avg_wait_variance() if result.n_rewards >= 0 else 0.0
                ),
                "rides_per_party": (
                    personal.rides_completed / max(personal.n_focals, 1)
                ),
                "must_do_completion_rate": personal.must_do_completion_rate,
                "avg_preference_score_per_guest": personal.avg_preference_score_per_guest,
                "avg_must_do_latency_sec": 0.0,
                "n_focals": personal.n_focals,
            }
        return obs, reward, terminated, truncated, info
