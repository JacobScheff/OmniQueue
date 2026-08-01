"""Load a trained ParkRouterModel checkpoint for evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from Park.model import forward_route_with_mask, obs_flat_to_tensors
from Park.training.checkpoint import load_checkpoint
from Park.training.route_reward import commit_action, pad_route


class PPOPolicy:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.step, self.meta = load_checkpoint(checkpoint, self.device)
        self.model.eval()
        self.route_k = int(getattr(self.model, "route_k", 6))

    @torch.no_grad()
    def act(self, obs_flat) -> int:
        route, _ = self.act_with_route(obs_flat)
        return commit_action(route)

    @torch.no_grad()
    def act_with_probs(self, obs_flat) -> tuple[int, np.ndarray]:
        """Return (commit action, masked softmax probabilities over slot-0)."""
        route, probs = self.act_with_route(obs_flat)
        return commit_action(route), probs

    @torch.no_grad()
    def act_with_route(self, obs_flat) -> tuple[np.ndarray, np.ndarray]:
        """Return (route[K], slot-0 masked softmax probabilities)."""
        obs = torch.tensor(
            np.asarray(obs_flat), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        guest, ride, env = obs_flat_to_tensors(obs)
        out = forward_route_with_mask(
            self.model, guest, ride, env, routes=None, deterministic=True
        )
        route = pad_route(out.routes[0].cpu().numpy().tolist(), self.route_k)
        probs = torch.softmax(out.slot0_logits[0], dim=-1).cpu().numpy()
        return route, probs

    @torch.no_grad()
    def act_batch(self, obs_batch) -> np.ndarray:
        actions, _ = self.act_batch_with_probs(obs_batch)
        return actions

    @torch.no_grad()
    def act_batch_with_probs(self, obs_batch) -> tuple[np.ndarray, np.ndarray]:
        """Return (commit actions [B], slot-0 probs [B, A])."""
        routes, probs = self.act_batch_with_routes(obs_batch)
        return np.asarray([commit_action(r) for r in routes], dtype=np.int64), probs

    @torch.no_grad()
    def act_batch_with_routes(self, obs_batch) -> tuple[np.ndarray, np.ndarray]:
        """Return (routes [B, K], slot-0 probs [B, A])."""
        obs = torch.tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        guest, ride, env = obs_flat_to_tensors(obs)
        out = forward_route_with_mask(
            self.model, guest, ride, env, routes=None, deterministic=True
        )
        routes = out.routes.cpu().numpy().astype(np.int64)
        probs = torch.softmax(out.slot0_logits, dim=-1).cpu().numpy()
        return routes, probs


class PPORouter:
    """Phase 3 router stub — online routing in the native sim is not yet wired."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu"):
        self.policy = PPOPolicy(checkpoint, device) if checkpoint else None

    def route_batch(self, party_ids, parties, rides, graph, now_sec, rng):
        raise NotImplementedError(
            "Online PPO routing in full-day sim is not wired yet. "
            "Train with training/ppo_train.py and evaluate with training/eval_policy.py."
        )
