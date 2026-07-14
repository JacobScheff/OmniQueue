#!/usr/bin/env python3
"""Interactive human-vs-AI park day player.

Usage:
    python play.py --seed 42 --checkpoint checkpoints/ppo/ppo_final.pt
    python play.py --seed 42 --crowd heuristic --speed 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive OmniQueue play mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="PPO checkpoint path (required for PPO crowd / AI compare / benchmark)",
    )
    parser.add_argument(
        "--crowd",
        choices=("heuristic", "ppo"),
        default="heuristic",
        help="Router for all non-you parties during play",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--speed", type=float, default=120.0, help="Sim seconds per real second")
    parser.add_argument("--sample-interval", type=int, default=60)
    args = parser.parse_args(argv)

    if args.crowd == "ppo":
        if not args.checkpoint:
            print("error: --checkpoint is required when --crowd ppo", file=sys.stderr)
            return 2
        if not Path(args.checkpoint).is_file():
            print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 2

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("error: pygame required. Install with: pip install -e '.[viz]'", file=sys.stderr)
        return 2

    from play.app import run_play_app

    run_play_app(
        seed=args.seed,
        checkpoint=args.checkpoint,
        crowd_router=args.crowd,
        device=args.device,
        speed=args.speed,
        sample_interval=args.sample_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
