#!/usr/bin/env python3
"""Evaluate a trained policy in the C++ ParkEnv (+ optional collapse probe)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PARENT = ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import numpy as np
import torch

from Park.model import forward_with_mask, obs_flat_to_tensors
from Park.router.ppo import PPOPolicy
from Park.training.env import ParkRoutingEnv
from Park.training.features import (
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    js_divergence,
    rewrite_prefs_must_dos,
    top_must_do_or_pref,
)
from Park.training.route_reward import commit_action


def collapse_probe(policy: PPOPolicy, template_obs: np.ndarray, *, n: int = 64) -> dict:
    """Fix waits/location; sweep random prefs/must-dos; report opener concentration."""
    device = policy.device
    base = torch.tensor(template_obs, dtype=torch.float32, device=device).unsqueeze(0)
    guest0, ride0, env0 = obs_flat_to_tensors(base)
    commits: list[int] = []
    js_vals: list[float] = []
    for _ in range(n):
        g, r = rewrite_prefs_must_dos(guest0, ride0)
        with torch.no_grad():
            out_logits, _, mask = forward_with_mask(policy.model, g, r, env0)
            probs = torch.softmax(out_logits.masked_fill(~mask, -1.0e9), dim=-1)
            commits.append(int(probs.argmax(dim=-1).item()))
        # Pairwise JS vs another rewrite
        g2, r2 = rewrite_prefs_must_dos(guest0, ride0)
        if int(top_must_do_or_pref(g, r)[0]) == int(top_must_do_or_pref(g2, r2)[0]):
            g2, r2 = rewrite_prefs_must_dos(guest0, ride0)
        with torch.no_grad():
            logits_b, _, mask_b = forward_with_mask(policy.model, g2, r2, env0)
            probs_b = torch.softmax(logits_b.masked_fill(~mask_b, -1.0e9), dim=-1)
            js_vals.append(float(js_divergence(probs, probs_b)[0].item()))

    counts = Counter(commits)
    top_ride, top_count = counts.most_common(1)[0]
    return {
        "n": n,
        "top1_action": int(top_ride),
        "top1_share": top_count / max(n, 1),
        "unique_openers": len(counts),
        "mean_pairwise_js": float(np.mean(js_vals)) if js_vals else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained routing policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--collapse-probe",
        type=int,
        default=64,
        help="Random pref sweeps for opener collapse probe (0=skip)",
    )
    args = parser.parse_args()

    policy = PPOPolicy(args.checkpoint, device=args.device)
    env = ParkRoutingEnv(seed=args.seed)
    template_obs = None

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if template_obs is None:
            template_obs = np.asarray(obs, dtype=np.float32).copy()
        total_reward = 0.0
        steps = 0
        while True:
            route, _ = policy.act_with_route(obs)
            action = commit_action(route)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                metrics = info.get("metrics", {})
                must_rate = metrics.get("must_do_completion_rate", float("nan"))
                pref = metrics.get("avg_preference_score_per_guest", float("nan"))
                latency_min = float(metrics.get("avg_must_do_latency_sec", 0.0) or 0.0) / 60.0
                print(
                    f"episode={ep + 1} steps={steps} reward={total_reward:.3f} "
                    f"rides={metrics.get('rides_completed', '?')} "
                    f"must_do={must_rate:.3f} pref/guest={pref:.4f} "
                    f"must_do_latency={latency_min:.1f}m "
                    f"wait_var={metrics.get('avg_wait_variance', 0):.1f} "
                    f"last_route={route.tolist()}"
                )
                break

    if args.collapse_probe > 0 and template_obs is not None:
        # Ensure ride feats look open for the probe template.
        g_end = GUEST_FEAT_DIM
        r_end = g_end + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
        ride = template_obs[g_end:r_end].reshape(NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
        ride[:, 2] = 1.0  # open
        ride[:, 5] = 0.05  # short walk
        template_obs[37] = max(float(template_obs[37]), 0.5)
        probe = collapse_probe(policy, template_obs, n=args.collapse_probe)
        print(
            f"collapse_probe: n={probe['n']} top1_action={probe['top1_action']} "
            f"top1_share={probe['top1_share']:.3f} unique={probe['unique_openers']} "
            f"mean_js={probe['mean_pairwise_js']:.4f}"
        )


if __name__ == "__main__":
    main()
