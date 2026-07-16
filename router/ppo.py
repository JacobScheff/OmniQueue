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

        actions, _ = self.act_with_probs(obs_flat)
        return int(actions)

    @torch.no_grad()
    def act_with_probs(self, obs_flat) -> tuple[int, "np.ndarray"]:
        """Return (argmax action, masked softmax probabilities) for one flat obs."""
        import numpy as np

        obs = torch.tensor(np.asarray(obs_flat), dtype=torch.float32, device=self.device).unsqueeze(0)
        guest, ride, env = obs_flat_to_tensors(obs)
        logits, _, _ = forward_with_mask(self.model, guest, ride, env)
        probs = torch.softmax(logits[0, 0], dim=-1)
        action = int(probs.argmax().item())
        return action, probs.cpu().numpy()

    @torch.no_grad()
    def act_batch(self, obs_batch) -> "np.ndarray":
        actions, _ = self.act_batch_with_probs(obs_batch)
        return actions

    @torch.no_grad()
    def act_batch_with_probs(self, obs_batch) -> tuple["np.ndarray", "np.ndarray"]:
        """Return (actions [G], probs [G, A]) for a co-timed PPO batch."""
        import numpy as np

        obs = torch.tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        # Co-timed batches use joint coordinator attention.
        guest, ride, env = obs_group_to_tensors(obs)
        logits, _, _ = forward_with_mask(self.model, guest, ride, env)
        probs = torch.softmax(logits[0], dim=-1)  # (G, A)
        actions = probs.argmax(dim=-1).cpu().numpy()
        return actions, probs.cpu().numpy()


class PPORouter:
    """Phase 3 router stub — online routing in the native sim is not yet wired."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu"):
        self.policy = PPOPolicy(checkpoint, device) if checkpoint else None

    def route_batch(self, party_ids, parties, rides, graph, now_sec, rng):
        raise NotImplementedError(
            "Online PPO routing in full-day sim is not wired yet. "
            "Train with training/ppo_train.py and evaluate with training/eval_policy.py."
        )
