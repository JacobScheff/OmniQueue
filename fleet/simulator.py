"""Python entry point for the C++ discrete-event fleet simulator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure the fleet package dir (where _fleet_sim is installed) is importable.
_FLEET_DIR = Path(__file__).resolve().parent
if str(_FLEET_DIR) not in sys.path:
    sys.path.insert(0, str(_FLEET_DIR))


def _require_native():
    try:
        import _fleet_sim  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "C++ extension _fleet_sim is not built. Run: pip install -e ."
        ) from exc
    return _fleet_sim


def native_backend_name() -> str:
    try:
        _require_native()
    except ImportError:
        return "unavailable"
    return "native"


def _fill_sim_config(cfg: Any, **fields: Any) -> Any:
    """Copy keyword fields onto a native RecordConfig / EnvConfig."""
    for key, value in fields.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _default_city_size(num_intersections: int) -> int:
    return max(800, int(num_intersections * 15))


def make_env_config(
    *,
    city_width: int | None = None,
    city_height: int | None = None,
    num_intersections: int = 80,
    num_vehicles: int = 30,
    num_requests: int = 120,
    horizon_sec: int = 3600,
    vehicle_speed: float = 2.0,
    vehicle_capacity: int = 1,
    avg_streets_per_intersection: int = 5,
):
    """Build a native ``EnvConfig`` (shared by PPO train / rollout)."""
    _fleet_sim = _require_native()
    width = city_width if city_width is not None else _default_city_size(num_intersections)
    height = city_height if city_height is not None else width
    return _fill_sim_config(
        _fleet_sim.EnvConfig(),
        city_width=width,
        city_height=height,
        num_intersections=num_intersections,
        num_vehicles=num_vehicles,
        num_requests=num_requests,
        horizon_sec=horizon_sec,
        vehicle_speed=vehicle_speed,
        vehicle_capacity=vehicle_capacity,
        avg_streets_per_intersection=avg_streets_per_intersection,
    )


def record_day(
    seed: int = 0,
    sample_interval_sec: int = 60,
    *,
    city_width: int = 1200,
    city_height: int = 1200,
    num_intersections: int = 80,
    num_vehicles: int = 30,
    num_requests: int = 3840,
    horizon_sec: int = 86400,
    vehicle_speed: float = 2.0,
    vehicle_capacity: int = 1,
    avg_streets_per_intersection: int = 5,
):
    """Simulate one heuristic fleet day/shift and return a ``DayRecording``."""
    _fleet_sim = _require_native()
    cfg = _fill_sim_config(
        _fleet_sim.RecordConfig(),
        city_width=city_width,
        city_height=city_height,
        num_intersections=num_intersections,
        num_vehicles=num_vehicles,
        num_requests=num_requests,
        horizon_sec=horizon_sec,
        vehicle_speed=vehicle_speed,
        vehicle_capacity=vehicle_capacity,
        avg_streets_per_intersection=avg_streets_per_intersection,
    )
    return _fleet_sim.record_day(seed, cfg, sample_interval_sec)


def _load_ppo_model(checkpoint: str | Path, device: str = "cpu"):
    """Load a ``VehicleRouter`` from a ``ppo_train`` checkpoint."""
    import torch

    from fleet.model import VehicleRouter

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"PPO checkpoint not found: {path}\n"
            f"Train one with: python -m fleet.training.ppo_train --seed 42"
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = VehicleRouter(use_graph_encoder=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def record_day_ppo(
    seed: int = 0,
    checkpoint: str | Path = "checkpoints/ppo/ppo_final.pt",
    sample_interval_sec: int = 60,
    *,
    device: str = "cpu",
    city_width: int | None = None,
    city_height: int | None = None,
    num_intersections: int = 80,
    num_vehicles: int = 30,
    num_requests: int = 120,
    horizon_sec: int = 3600,
    vehicle_speed: float = 2.0,
    vehicle_capacity: int = 1,
    max_steps: int = 50_000,
):
    """Roll out a PPO policy for one episode and return a ``DayRecording``."""
    import numpy as np
    import torch

    import fleet.config as config
    from fleet.model import forward_with_mask

    _fleet_sim = _require_native()
    model, ckpt = _load_ppo_model(checkpoint, device=device)
    saved = ckpt.get("config") or {}
    num_vehicles = int(saved.get("num_vehicles", num_vehicles))
    num_requests = int(saved.get("num_requests", num_requests))
    num_intersections = int(saved.get("num_intersections", num_intersections))
    horizon_sec = int(saved.get("horizon_sec", horizon_sec))
    avg_streets = int(
        saved.get(
            "avg_streets_per_intersection",
            5,
        )
    )

    env = _fleet_sim.FleetEnv(
        seed,
        make_env_config(
            city_width=city_width,
            city_height=city_height,
            num_intersections=num_intersections,
            num_vehicles=num_vehicles,
            num_requests=num_requests,
            horizon_sec=horizon_sec,
            vehicle_speed=vehicle_speed,
            vehicle_capacity=vehicle_capacity,
            avg_streets_per_intersection=avg_streets,
        ),
    )
    env.enable_recording(sample_interval_sec)
    obs = env.reset(seed)
    flat = np.asarray(obs.flat(), dtype=np.float32)
    if flat.shape[0] != config.FLAT_OBS_DIM:
        raise RuntimeError(
            f"FLAT_OBS_DIM mismatch: python={config.FLAT_OBS_DIM} "
            f"native={flat.shape[0]} (rebuild _fleet_sim)"
        )

    steps = 0
    with torch.no_grad():
        while steps < max_steps:
            obs_t = torch.from_numpy(flat).unsqueeze(0).to(device)
            logits, _ = forward_with_mask(model, obs_t)
            action = int(logits[0, 0].argmax().item())
            result = env.step(action)
            steps += 1
            if result.done or not result.has_obs:
                break
            flat = np.asarray(result.obs.flat(), dtype=np.float32)

    return env.recording
