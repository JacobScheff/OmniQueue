"""Checkpoint helpers for BC and PPO training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import Park.config as config
from Park.model import RankRouteModel
from Park.training.features import (
    CANDIDATE_M,
    D_MODEL,
    ENV_DYNAMIC_FEAT_DIM,
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
)


@dataclass
class TrainConfig:
    guest_feat_dim: int = GUEST_FEAT_DIM
    num_rides: int = NUM_RIDES
    ride_dynamic_feat_dim: int = RIDE_DYNAMIC_FEAT_DIM
    environment_dynamic_feat_dim: int = ENV_DYNAMIC_FEAT_DIM
    d_model: int = D_MODEL
    route_k: int = 5
    candidate_m: int = CANDIDATE_M
    arch_version: str = "rank_route_v1"

    def __post_init__(self) -> None:
        self.route_k = int(getattr(config, "PPO_ROUTE_K", self.route_k))
        self.candidate_m = int(getattr(config, "PPO_CANDIDATE_M", self.candidate_m))
        self.arch_version = str(getattr(config, "MODEL_ARCH_VERSION", self.arch_version))


def default_model(device: str | torch.device = "cpu") -> RankRouteModel:
    cfg = TrainConfig()
    model = RankRouteModel(
        guest_feat_dim=cfg.guest_feat_dim,
        num_rides=cfg.num_rides,
        ride_dynamic_feat_dim=cfg.ride_dynamic_feat_dim,
        environment_dynamic_feat_dim=cfg.environment_dynamic_feat_dim,
        d_model=cfg.d_model,
        route_k=cfg.route_k,
        candidate_m=cfg.candidate_m,
    )
    return model.to(device)


def _load_state_flexible(model: RankRouteModel, state: dict) -> list[str]:
    """Load matching tensors; skip incompatible keys (no legacy widen bridges)."""
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    notes: list[str] = []
    skipped = [k for k in state if k not in filtered]
    if skipped:
        notes.append(f"skipped_incompatible={len(skipped)}")
    if missing:
        notes.append(f"missing={len(missing)}")
    if unexpected:
        notes.append(f"unexpected={len(unexpected)}")
    return notes


def save_checkpoint(
    path: str | Path,
    model: RankRouteModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig()
    meta_extra = {
        "arch_version": cfg.arch_version,
        "route_k": cfg.route_k,
        "candidate_m": cfg.candidate_m,
        **(extra or {}),
    }
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "extra": meta_extra,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"step": step, "path": str(path), **meta_extra}, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    *,
    strict: bool = False,
) -> tuple[RankRouteModel, int, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg_raw = payload.get("config", asdict(TrainConfig()))
    route_k = int(cfg_raw.get("route_k", getattr(config, "PPO_ROUTE_K", 5)))
    candidate_m = int(cfg_raw.get("candidate_m", getattr(config, "PPO_CANDIDATE_M", CANDIDATE_M)))
    model = RankRouteModel(
        guest_feat_dim=cfg_raw["guest_feat_dim"],
        num_rides=cfg_raw["num_rides"],
        ride_dynamic_feat_dim=cfg_raw["ride_dynamic_feat_dim"],
        environment_dynamic_feat_dim=cfg_raw["environment_dynamic_feat_dim"],
        d_model=cfg_raw.get("d_model", D_MODEL),
        route_k=route_k,
        candidate_m=candidate_m,
    )
    state = payload["model_state_dict"]
    extra = dict(payload.get("extra", {}) or {})
    if strict:
        model.load_state_dict(state)
    else:
        notes = _load_state_flexible(model, state)
        if notes:
            extra["load_notes"] = ", ".join(notes)
    model.to(device)
    if optimizer is not None and "optimizer_state_dict" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except (ValueError, RuntimeError):
            extra["optimizer_load"] = "skipped_incompatible"
    return model, int(payload.get("step", 0)), extra
