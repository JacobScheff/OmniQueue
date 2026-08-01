"""Gymnasium wrapper around the C++ ParkEnv (personal planner: 1 focal + heuristic crowd)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import _park_sim
from Park.training.features import FLAT_OBS_DIM, NUM_ACTIONS, route_k
from Park.training.route_reward import ROUTE_PAD, commit_action, pad_route


class ParkRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, num_focals: int = 1):
        super().__init__()
        self._env = _park_sim.ParkEnv(seed)
        self._seed = seed
        self._num_focals = int(num_focals)
        self._k = route_k()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FLAT_OBS_DIM,), dtype=np.float32
        )
        # Commit action remains Discrete for gym callers; routes are MultiDiscrete.
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.route_space = spaces.MultiDiscrete(
            [NUM_ACTIONS] + [NUM_ACTIONS] * (self._k - 1)
        )

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
        result = self._env.exchange_batch([], 1)
        if result.n_obs <= 0:
            obs = np.zeros(FLAT_OBS_DIM, dtype=np.float32)
        else:
            obs = np.asarray(result.obs, dtype=np.float32).reshape(FLAT_OBS_DIM)
        return obs, {}

    def step(self, action):
        """Accept a commit action int or a full route sequence; DES gets route[0]."""
        if isinstance(action, (list, tuple, np.ndarray)):
            route = pad_route(np.asarray(action, dtype=np.int64).tolist(), self._k)
            commit = commit_action(route)
        else:
            commit = int(action)
            route = pad_route([commit], self._k)

        result = self._env.exchange_batch([commit], 1)
        obs = np.zeros(FLAT_OBS_DIM, dtype=np.float32)
        reward = 0.0
        if result.n_rewards > 0:
            reward = float(np.asarray(result.rewards, dtype=np.float32)[0])
        if result.n_obs > 0:
            obs = np.asarray(result.obs, dtype=np.float32).reshape(-1, FLAT_OBS_DIM)[0]
        terminated = bool(result.episode_done)
        truncated = False
        info = {"route": route, "commit": commit, "route_pad": ROUTE_PAD}
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
