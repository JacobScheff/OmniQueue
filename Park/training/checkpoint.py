"""Checkpoint helpers for BC and PPO training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from Park.model import ParkRouterModel
from Park.training.features import (
    D_MODEL,
    ENV_DYNAMIC_FEAT_DIM,
    GUEST_FEAT_DIM,
    NUM_ATTN_HEADS,
    NUM_RIDES,
    NUM_TRANSFORMER_LAYERS,
    RIDE_DYNAMIC_FEAT_DIM,
)


@dataclass
class TrainConfig:
    guest_feat_dim: int = GUEST_FEAT_DIM
    num_rides: int = NUM_RIDES
    ride_dynamic_feat_dim: int = RIDE_DYNAMIC_FEAT_DIM
    environment_dynamic_feat_dim: int = ENV_DYNAMIC_FEAT_DIM
    d_model: int = D_MODEL
    num_layers: int = NUM_TRANSFORMER_LAYERS
    num_heads: int = NUM_ATTN_HEADS


def default_model(device: str | torch.device = "cpu") -> ParkRouterModel:
    cfg = TrainConfig()
    model = ParkRouterModel(
        guest_feat_dim=cfg.guest_feat_dim,
        num_rides=cfg.num_rides,
        ride_dynamic_feat_dim=cfg.ride_dynamic_feat_dim,
        environment_dynamic_feat_dim=cfg.environment_dynamic_feat_dim,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
    )
    return model.to(device)


def save_checkpoint(
    path: str | Path,
    model: ParkRouterModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": asdict(TrainConfig()),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"step": step, "path": str(path), **(extra or {})}, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[ParkRouterModel, int, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("config", asdict(TrainConfig()))
    model = ParkRouterModel(
        guest_feat_dim=cfg["guest_feat_dim"],
        num_rides=cfg["num_rides"],
        ride_dynamic_feat_dim=cfg["ride_dynamic_feat_dim"],
        environment_dynamic_feat_dim=cfg["environment_dynamic_feat_dim"],
        d_model=cfg.get("d_model", D_MODEL),
        num_layers=cfg.get("num_layers", NUM_TRANSFORMER_LAYERS),
        num_heads=cfg.get("num_heads", NUM_ATTN_HEADS),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, int(payload.get("step", 0)), payload.get("extra", {})
