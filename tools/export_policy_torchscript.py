#!/usr/bin/env python3
"""Export a ParkRouterModel checkpoint to TorchScript for C++ LibTorch inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.features import FLAT_OBS_DIM, NUM_ACTIONS
from training.policy_export import export_torchscript


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to TorchScript for native rollout")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    path = export_torchscript(args.checkpoint, output, args.device)
    print(f"Exported TorchScript policy: {path}  obs_dim={FLAT_OBS_DIM}  actions={NUM_ACTIONS}")


if __name__ == "__main__":
    main()
