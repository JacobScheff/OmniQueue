#!/usr/bin/env python3
"""Evaluate a trained policy in the C++ ParkEnv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.env import ParkRoutingEnv
from router.ppo import PPOPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained routing policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    policy = PPOPolicy(args.checkpoint, device=args.device)
    env = ParkRoutingEnv(seed=args.seed)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0
        while True:
            action = policy.act(obs)
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
                    f"wait_var={metrics.get('avg_wait_variance', 0):.1f}"
                )
                break


if __name__ == "__main__":
    main()
