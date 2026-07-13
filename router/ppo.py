"""Load a trained ParkRouterModel checkpoint for evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from model import forward_with_mask, obs_flat_to_tensors, obs_group_to_tensors
from training.checkpoint import load_checkpoint


class PPOPolicy:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.step, self.meta = load_checkpoint(checkpoint, self.device)
        self.model.eval()

    @torch.no_grad()
    def act(self, obs_flat) -> int:
        import numpy as np

        obs = torch.tensor(np.asarray(obs_flat), dtype=torch.float32, device=self.device).unsqueeze(0)
        guest, ride, env = obs_flat_to_tensors(obs)
        logits, _, _ = forward_with_mask(self.model, guest, ride, env)
        return int(logits[0, 0, :].argmax().item())

    @torch.no_grad()
    def act_batch(self, obs_batch) -> "np.ndarray":
        import numpy as np

        obs = torch.tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        # Co-timed batches use joint coordinator attention.
        guest, ride, env = obs_group_to_tensors(obs)
        logits, _, _ = forward_with_mask(self.model, guest, ride, env)
        return logits[0].argmax(dim=-1).cpu().numpy()


class PPORouter:
    """Phase 3 router stub — online routing in the native sim is not yet wired."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu"):
        self.policy = PPOPolicy(checkpoint, device) if checkpoint else None

    def route_batch(self, party_ids, parties, rides, graph, now_sec, rng):
        raise NotImplementedError(
            "Online PPO routing in full-day sim is not wired yet. "
            "Train with training/ppo_train.py and evaluate with training/eval_policy.py."
        )
