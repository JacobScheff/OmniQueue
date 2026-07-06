"""Load a trained ParkRouterModel checkpoint for evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model import ParkRouterModel, obs_flat_to_tensors
from training.checkpoint import load_checkpoint
from training.features import FLAT_OBS_DIM, NUM_ACTIONS


class PPOPolicy:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.step, self.meta = load_checkpoint(checkpoint, self.device)
        self.model.eval()

    @torch.no_grad()
    def act(self, obs_flat: np.ndarray) -> int:
        obs = torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        guest, ride, env = obs_flat_to_tensors(obs)
        logits, _ = self.model(guest, ride, env)
        return int(logits[0, 0, :].argmax().item())

    @torch.no_grad()
    def act_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        guest, ride, env = obs_flat_to_tensors(obs)
        logits, _ = self.model(guest, ride, env)
        return logits[:, 0, :].argmax(dim=-1).cpu().numpy()


class PPORouter:
    """Phase 3 router stub for integration tests."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu"):
        self.policy = PPOPolicy(checkpoint, device) if checkpoint else None

    def route_batch(self, party_ids, parties, rides, graph, now_sec, rng):
        raise NotImplementedError(
            "Online PPO routing in full-day sim is not wired yet. "
            "Train with training/ppo_train.py and evaluate with training/eval_policy.py."
        )
