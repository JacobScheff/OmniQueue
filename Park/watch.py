#!/usr/bin/env python3
"""PPO-focal park day watcher (timeline + mid-day preference edits).

Usage:
    python watch.py --model checkpoints/ppo/ppo_final.pt
    python watch.py --seed 42 --model checkpoints/ppo/ppo_final.pt --speed 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OmniQueue watch mode (PPO focal guest)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        type=str,
        required=True,
        help="PPO model/checkpoint file path (shared by focal and optional PPO crowd)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--speed", type=float, default=120.0, help="Sim seconds per real second")
    parser.add_argument("--sample-interval", type=int, default=60)
    parser.add_argument(
        "--crowd",
        type=str,
        default="heuristic",
        choices=("heuristic", "ppo"),
        help="Initial background guest router (also toggleable in Setup UI)",
    )
    args = parser.parse_args(argv)

    if not Path(args.model).is_file():
        print(f"error: PPO model not found: {args.model}", file=sys.stderr)
        return 2

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("error: pygame required. Install with: pip install -e '.[viz]'", file=sys.stderr)
        return 2

    from watch.app import run_watch_app

    run_watch_app(
        seed=args.seed,
        checkpoint=args.model,
        crowd_router=args.crowd,
        device=args.device,
        speed=args.speed,
        sample_interval=args.sample_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
