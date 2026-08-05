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


def _infer_d_model_from_state(state: dict) -> int | None:
    """Read d_model from checkpoint weights (authoritative over possibly-stale config)."""
    w = state.get("ride_id_embed.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[1])
    return None


def _config_from_model(model: RankRouteModel) -> TrainConfig:
    """Build TrainConfig matching a live model's architecture."""
    cfg = TrainConfig()
    cfg.d_model = int(model.d_model)
    cfg.route_k = int(model.route_k)
    cfg.candidate_m = int(model.candidate_m)
    cfg.num_rides = int(model.num_rides)
    cfg.environment_dynamic_feat_dim = int(model.env_dim)
    try:
        cfg.ride_dynamic_feat_dim = int(model.ride_feat_proj[0].in_features)
    except Exception:
        pass
    try:
        cfg.guest_feat_dim = int(model.guest_proj[0].in_features) - int(model.env_dim)
    except Exception:
        pass
    return cfg


def _widen_guest_proj_state(
    model: RankRouteModel,
    state: dict,
    *,
    env_dim: int = ENV_DYNAMIC_FEAT_DIM,
) -> tuple[dict, list[str]]:
    """Widen guest_proj.0.weight when checkpoint guest dim is smaller than current.

    Layout of Linear in-features: [guest_0.. | env_0..]. New guest columns (appended
    before env) are zero-initialized so the prior mapping stays intact at load.
    """
    notes: list[str] = []
    key = "guest_proj.0.weight"
    if key not in state or key not in model.state_dict():
        return state, notes
    old_w = state[key]
    new_w = model.state_dict()[key]
    if tuple(old_w.shape) == tuple(new_w.shape):
        return state, notes
    if old_w.ndim != 2 or new_w.ndim != 2:
        return state, notes
    if old_w.shape[0] != new_w.shape[0]:
        return state, notes
    old_in = int(old_w.shape[1])
    new_in = int(new_w.shape[1])
    if old_in >= new_in or old_in <= env_dim or new_in <= env_dim:
        return state, notes
    old_guest = old_in - env_dim
    new_guest = new_in - env_dim
    if new_guest <= old_guest:
        return state, notes
    widened = new_w.detach().cpu().clone()
    widened.zero_()
    widened[:, :old_guest] = old_w[:, :old_guest]
    widened[:, new_guest:] = old_w[:, old_guest:]
    out = dict(state)
    out[key] = widened.to(dtype=old_w.dtype)
    notes.append(f"widened_guest_proj {old_in}->{new_in}")
    return out, notes


def _load_state_flexible(model: RankRouteModel, state: dict) -> list[str]:
    """Load matching tensors; skip incompatible keys (after optional guest widen)."""
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
    # Persist the *model's* architecture — not the process-wide TrainConfig defaults
    # (e.g. a d_model=384 checkpoint must not be re-saved as d_model=512).
    cfg = _config_from_model(model)
    meta_extra = {
        "arch_version": cfg.arch_version,
        "route_k": cfg.route_k,
        "candidate_m": cfg.candidate_m,
        "guest_feat_dim": cfg.guest_feat_dim,
        "ride_dynamic_feat_dim": cfg.ride_dynamic_feat_dim,
        "d_model": cfg.d_model,
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
    cfg = TrainConfig()
    cfg_raw = payload.get("config", asdict(cfg))
    state = payload["model_state_dict"]
    # Guest feat dim: current code (widen bridge). d_model: weights win over stale config
    # (BC once saved d_model=512 metadata around a 384-weight file).
    inferred_d = _infer_d_model_from_state(state)
    d_model = int(inferred_d if inferred_d is not None else cfg_raw.get("d_model", cfg.d_model))
    route_k = int(cfg_raw.get("route_k", cfg.route_k))
    candidate_m = int(cfg_raw.get("candidate_m", cfg.candidate_m))
    ride_dim = int(cfg_raw.get("ride_dynamic_feat_dim", cfg.ride_dynamic_feat_dim))
    model = RankRouteModel(
        guest_feat_dim=cfg.guest_feat_dim,
        num_rides=cfg.num_rides,
        ride_dynamic_feat_dim=ride_dim,
        environment_dynamic_feat_dim=cfg.environment_dynamic_feat_dim,
        d_model=d_model,
        route_k=route_k,
        candidate_m=candidate_m,
    )
    extra = dict(payload.get("extra", {}) or {})
    if inferred_d is not None and int(cfg_raw.get("d_model", d_model)) != d_model:
        extra["d_model_from_weights"] = d_model
    state, widen_notes = _widen_guest_proj_state(model, state)
    if strict:
        model.load_state_dict(state)
    else:
        notes = widen_notes + _load_state_flexible(model, state)
        if notes:
            extra["load_notes"] = ", ".join(notes)
    model.to(device)
    if optimizer is not None and "optimizer_state_dict" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except (ValueError, RuntimeError):
            extra["optimizer_load"] = "skipped_incompatible"
    return model, int(payload.get("step", 0)), extra
