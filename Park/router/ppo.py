"""Load a trained RankRouteModel checkpoint for evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import Park.config as config
from Park.model import forward_route_with_mask, obs_flat_to_tensors
from Park.training.checkpoint import load_checkpoint
from Park.training.route_reward import commit_action, pad_route


class PPOPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        *,
        close_margin: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        self.device = torch.device(device)
        self.model, self.step, self.meta = load_checkpoint(checkpoint, self.device)
        self.model.eval()
        self.route_k = int(getattr(self.model, "route_k", 5))
        self.close_margin = float(
            close_margin
            if close_margin is not None
            else getattr(config, "INFER_CLOSE_MARGIN", 0.12)
        )
        self.temperature = float(
            temperature if temperature is not None else getattr(config, "INFER_TEMP", 0.8)
        )
        self.top_p = float(
            top_p if top_p is not None else getattr(config, "INFER_TOP_P", 0.9)
        )

    def _infer_kwargs(self, *, deterministic: bool) -> dict:
        if deterministic:
            # Close-call sampling still allowed when margin > 0
            return {
                "deterministic": True,
                "temperature": self.temperature,
                "close_margin": self.close_margin,
                "top_p": self.top_p,
            }
        return {
            "deterministic": False,
            "temperature": self.temperature,
            "close_margin": 0.0,
            "top_p": self.top_p,
        }

    @torch.no_grad()
    def act(self, obs_flat) -> int:
        route, _ = self.act_with_route(obs_flat)
        return commit_action(route)

    @torch.no_grad()
    def act_with_probs(self, obs_flat) -> tuple[int, np.ndarray]:
        """Return (commit action, masked softmax probabilities over Stage A)."""
        route, probs = self.act_with_route(obs_flat)
        return commit_action(route), probs

    @torch.no_grad()
    def act_with_route(self, obs_flat) -> tuple[np.ndarray, np.ndarray]:
        """Return (route[K], Stage A masked softmax probabilities)."""
        obs = torch.tensor(
            np.asarray(obs_flat), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        guest, ride, env = obs_flat_to_tensors(obs)
        out = forward_route_with_mask(
            self.model,
            guest,
            ride,
            env,
            routes=None,
            **self._infer_kwargs(deterministic=True),
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
        """Return (commit actions [B], Stage A probs [B, A])."""
        routes, probs = self.act_batch_with_routes(obs_batch)
        return np.asarray([commit_action(r) for r in routes], dtype=np.int64), probs

    @torch.no_grad()
    def act_batch_with_routes(self, obs_batch) -> tuple[np.ndarray, np.ndarray]:
        """Return (routes [B, K], Stage A probs [B, A])."""
        obs = torch.tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        guest, ride, env = obs_flat_to_tensors(obs)
        out = forward_route_with_mask(
            self.model,
            guest,
            ride,
            env,
            routes=None,
            **self._infer_kwargs(deterministic=True),
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
