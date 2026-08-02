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
    GUEST_FEAT_TIME_LEFT,
    NUM_RIDES,
)


class _CompanionExportWrapper(torch.nn.Module):
    """Companion forward: guest/ride/env/force_first → route + slot logits."""

    def __init__(self, model: ParkRouterModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        guest: torch.Tensor,
        ride: torch.Tensor,
        env: torch.Tensor,
        force_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model.forward_route(
            guest,
            ride,
            env,
            routes=None,
            deterministic=True,
            force_first=force_first,
        )
        return (
            out.routes.to(dtype=torch.int64),
            out.slot0_logits,
            out.slot_logits,
            out.slot_masks.to(dtype=torch.float32),
        )


def export_one(pt_path: Path, onnx_path: Path, *, opset: int) -> None:
    model, step, extra = load_checkpoint(pt_path, device="cpu")
    model.eval()
    wrapped = _CompanionExportWrapper(model)
    wrapped.eval()
    ride_feat_dim = int(model.ride_feat_proj[0].in_features)

    guest = torch.zeros(1, GUEST_FEAT_DIM, dtype=torch.float32)
    # Mid-day open park so slot-0 can pick a ride; soft-close (time_left=0)
    # would make exit-only and drop the ride-update path from the ONNX graph
    # under legacy torch.onnx tracing.
    guest[..., GUEST_FEAT_TIME_LEFT] = 0.5
    guest[..., :NUM_RIDES] = 1.0 / float(NUM_RIDES)
    ride = torch.zeros(1, NUM_RIDES, ride_feat_dim, dtype=torch.float32)
    # Open rides so the route decoder's empty-mask fallback is traced into ONNX.
    ride[..., 2] = 1.0
    ride[..., 5] = 0.1
    env = torch.zeros(1, ENV_DYNAMIC_FEAT_DIM, dtype=torch.float32)
    # Export with no-force; force branch still traced via always-on tensor ops.
    force_first = torch.full((1,), -1, dtype=torch.int64)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapped,
            (guest, ride, env, force_first),
            str(onnx_path),
            input_names=["guest", "ride", "env", "force_first"],
            output_names=["route", "slot0_logits", "slot_logits", "slot_masks"],
            opset_version=opset,
            dynamo=False,
        )

    # Smoke: ORT vs torch
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    g = rng.standard_normal((1, GUEST_FEAT_DIM), dtype=np.float32)
    g[..., GUEST_FEAT_TIME_LEFT] = 0.5
    g[..., :NUM_RIDES] = 1.0 / float(NUM_RIDES)
    r = rng.standard_normal((1, NUM_RIDES, ride_feat_dim), dtype=np.float32)
    e = rng.standard_normal((1, ENV_DYNAMIC_FEAT_DIM), dtype=np.float32)
    # Open all rides so masks are non-degenerate for the smoke compare.
    r[..., 2] = 1.0
    r[..., 5] = 0.1
    ff_none = np.asarray([-1], dtype=np.int64)
    feeds = {"guest": g, "ride": r, "env": e, "force_first": ff_none}
    ort_route, ort_logits, ort_slot_logits, ort_slot_masks = sess.run(None, feeds)
    with torch.inference_mode():
        torch_route, torch_logits, torch_slot_logits, torch_slot_masks = wrapped(
            torch.from_numpy(g),
            torch.from_numpy(r),
            torch.from_numpy(e),
            torch.from_numpy(ff_none),
        )
        torch_route = torch_route.numpy()
        torch_logits = torch_logits.numpy()
        torch_slot_logits = torch_slot_logits.numpy()
        torch_slot_masks = torch_slot_masks.numpy()
    max_abs = float(np.max(np.abs(ort_logits - torch_logits)))
    slot_max_abs = float(np.max(np.abs(ort_slot_logits - torch_slot_logits)))
    route_match = bool(np.array_equal(ort_route, torch_route))
    ort_rides = [int(x) for x in np.asarray(ort_route).reshape(-1).tolist() if int(x) >= 0]
    unique_rides = len(ort_rides) == len(set(ort_rides))

    # Force-first smoke: pin a legal ride that is not the natural opener when possible.
    natural0 = int(ort_rides[0]) if ort_rides else 0
    force_id = 0 if natural0 != 0 else 1
    ff_force = np.asarray([force_id], dtype=np.int64)
    ort_forced_route, _, _, _ = sess.run(
        None, {"guest": g, "ride": r, "env": e, "force_first": ff_force}
    )
    forced_rides = [
        int(x) for x in np.asarray(ort_forced_route).reshape(-1).tolist() if int(x) >= 0
    ]
    force_ok = bool(forced_rides) and forced_rides[0] == force_id
    force_unique = len(forced_rides) == len(set(forced_rides))

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    meta = {
        "step": int(step),
        "path": str(onnx_path),
        "source_pt": str(pt_path),
        "arch_version": "rank_route_v1",
        "route_k": int(model.route_k),
        "candidate_m": int(getattr(model, "candidate_m", 8)),
        "ride_dynamic_feat_dim": ride_feat_dim,
        **{k: v for k, v in (extra or {}).items() if isinstance(v, (str, int, float, bool))},
    }
    onnx_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(
        f"OK {pt_path.name} -> {onnx_path.name} "
        f"(step={step}, {size_mb:.1f} MiB, max_abs_delta={max_abs:.3e}, "
        f"slot_max_abs={slot_max_abs:.3e}, route_match={route_match}, "
        f"unique_rides={unique_rides}, force_ok={force_ok}, "
        f"stub={bool(extra.get('stub'))})"
    )
    if max_abs > 1e-4 or slot_max_abs > 1e-4:
        raise SystemExit(
            f"ONNX mismatch too large for {pt_path}: "
            f"slot0={max_abs}, slots={slot_max_abs}"
        )
    if not route_match:
        raise SystemExit(f"ONNX route decode mismatch for {pt_path}: {ort_route} vs {torch_route}")
    if len(ort_rides) > 1 and not unique_rides:
        raise SystemExit(
            f"ONNX route repeats rides for open-park smoke ({pt_path}): {ort_rides}"
        )
    if not force_ok:
        raise SystemExit(
            f"ONNX force_first failed for {pt_path}: wanted {force_id}, got {forced_rides}"
        )
    if len(forced_rides) > 1 and not force_unique:
        raise SystemExit(
            f"ONNX forced route repeats rides ({pt_path}): {forced_rides}"
        )
    # slot_masks should be float 0/1; ignore unused exit/idle on tail slots.
    if ort_slot_masks.shape != torch_slot_masks.shape:
        raise SystemExit(
            f"ONNX slot_masks shape mismatch for {pt_path}: "
            f"{ort_slot_masks.shape} vs {torch_slot_masks.shape}"
        )


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
