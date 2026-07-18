#!/usr/bin/env python3
"""Interactive human-vs-AI park day player.

Usage:
    python play.py --seed 42
    python play.py --seed 42 --model checkpoints/ppo/ppo_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Parent of the Park/ package dir must be on sys.path for `import Park.*`.
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive OmniQueue play mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        type=str,
        default=None,
        help="PPO model/checkpoint file path (used when crowd is PPO, AI compare, or benchmark)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--speed", type=float, default=120.0, help="Sim seconds per real second")
    parser.add_argument("--sample-interval", type=int, default=60)
    args = parser.parse_args(argv)

    if args.model is not None and not Path(args.model).is_file():
        print(f"error: PPO model not found: {args.model}", file=sys.stderr)
        return 2

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("error: pygame required. Install with: pip install -e '.[viz]'", file=sys.stderr)
        return 2

    from Park.play.app import run_play_app

    run_play_app(
        seed=args.seed,
        checkpoint=args.model,
        device=args.device,
        speed=args.speed,
        sample_interval=args.sample_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
