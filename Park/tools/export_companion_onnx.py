#!/usr/bin/env python3
"""Export companion PPO checkpoints to ONNX for free-tier CPU hosting.

Usage (from repo root):
    PYTHONPATH=. python Park/tools/export_companion_onnx.py

Writes sibling .onnx files next to each configured .pt under companion/model/.
You only need this when a checkpoint changes; committed .onnx files are enough to deploy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Repo root on path when run as a script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Park.companion import settings
from Park.model import ParkRouterModel
from Park.training.checkpoint import load_checkpoint
from Park.training.features import (
    ENV_DYNAMIC_FEAT_DIM,
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
)


class _CompanionExportWrapper(torch.nn.Module):
    """Single-party companion forward: guest/ride/env → action logits."""

    def __init__(self, model: ParkRouterModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        guest: torch.Tensor,
        ride: torch.Tensor,
        env: torch.Tensor,
    ) -> torch.Tensor:
        # guest: (1, GUEST), ride: (1, R, F), env: (1, E)
        logits, _values = self.model(guest, ride, env)
        return logits


def export_one(pt_path: Path, onnx_path: Path, *, opset: int) -> None:
    model, step, extra = load_checkpoint(pt_path, device="cpu")
    model.eval()
    wrapped = _CompanionExportWrapper(model)
    wrapped.eval()

    guest = torch.zeros(1, GUEST_FEAT_DIM, dtype=torch.float32)
    ride = torch.zeros(1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM, dtype=torch.float32)
    env = torch.zeros(1, ENV_DYNAMIC_FEAT_DIM, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapped,
            (guest, ride, env),
            str(onnx_path),
            input_names=["guest", "ride", "env"],
            output_names=["logits"],
            opset_version=opset,
            dynamo=False,
        )

    # Smoke: ORT vs torch
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    g = rng.standard_normal((1, GUEST_FEAT_DIM), dtype=np.float32)
    r = rng.standard_normal((1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM), dtype=np.float32)
    e = rng.standard_normal((1, ENV_DYNAMIC_FEAT_DIM), dtype=np.float32)
    ort_out = sess.run(None, {"guest": g, "ride": r, "env": e})[0]
    with torch.inference_mode():
        torch_out = wrapped(
            torch.from_numpy(g),
            torch.from_numpy(r),
            torch.from_numpy(e),
        ).numpy()
    max_abs = float(np.max(np.abs(ort_out - torch_out)))
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    meta = {
        "step": int(step),
        "path": str(onnx_path),
        "source_pt": str(pt_path),
        **{k: v for k, v in (extra or {}).items() if isinstance(v, (str, int, float, bool))},
    }
    onnx_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(
        f"OK {pt_path.name} → {onnx_path.name} "
        f"(step={step}, {size_mb:.1f} MiB, max|Δ|={max_abs:.3e}, stub={bool(extra.get('stub'))})"
    )
    if max_abs > 1e-4:
        raise SystemExit(f"ONNX mismatch too large for {pt_path}: {max_abs}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default 17)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Export only this settings.MODELS key (repeatable)",
    )
    args = parser.parse_args()

    keys = args.only or list(settings.MODELS)
    exported = 0
    for key in keys:
        configured = Path(settings.MODELS[key])
        # Settings may already point at .onnx for deploy; prefer sibling .pt as export source.
        if configured.suffix.lower() == ".onnx":
            pt_path = configured.with_suffix(".pt")
            onnx_path = configured
        else:
            pt_path = configured
            onnx_path = configured.with_suffix(".onnx")
        if not pt_path.is_file():
            print(f"skip {key}: missing {pt_path}")
            continue
        export_one(pt_path, onnx_path, opset=args.opset)
        exported += 1
    if exported == 0:
        raise SystemExit("No checkpoints exported (place .pt files under companion/model/).")


if __name__ == "__main__":
    main()
