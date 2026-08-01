"""Load PPO checkpoints (ONNX preferred) and run single-party live inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from Park.companion import settings
from Park.companion.server.obs import ACTION_LABELS, action_label
from Park.training.features import (
    CLOSE_DRAIN_SEC,
    DAY_SECONDS,
    ENV_DYNAMIC_FEAT_DIM,
    FLAT_OBS_DIM,
    GUEST_FEAT_DIM,
    GUEST_FEAT_AT_RIDE_NODE,
    GUEST_FEAT_TIME_LEFT,
    NUM_ACTIONS,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    RIDE_FEAT_DURATION,
    RIDE_FEAT_OPEN,
    RIDE_FEAT_WAIT,
    RIDE_FEAT_WALK,
)

logger = logging.getLogger(__name__)

# Dedicated path for auto-generated random weights — never overwrite configured models.
_STUB_ONNX = Path(__file__).resolve().parents[1] / "model" / "_stub_random.onnx"
_STUB_PT = Path(__file__).resolve().parents[1] / "model" / "_stub_random.pt"


def _split_flat_obs(obs_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """flat [FLAT_OBS_DIM] → guest (1,F), ride (1,R,F), env (1,E)."""
    flat = np.asarray(obs_flat, dtype=np.float32).reshape(-1)
    if flat.shape[0] != FLAT_OBS_DIM:
        raise ValueError(f"obs_flat must have length {FLAT_OBS_DIM}, got {flat.shape[0]}")
    guest_end = GUEST_FEAT_DIM
    ride_end = guest_end + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    guest = flat[:guest_end].reshape(1, GUEST_FEAT_DIM)
    ride = flat[guest_end:ride_end].reshape(1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = flat[ride_end:].reshape(1, ENV_DYNAMIC_FEAT_DIM)
    return guest, ride, env


def build_action_mask_numpy(
    guest: np.ndarray,
    ride: np.ndarray,
    env: np.ndarray,
) -> np.ndarray:
    """Boolean mask (B, A) matching training.features.build_action_mask."""
    batch, num_rides, _ = ride.shape

    open_ok = ride[..., RIDE_FEAT_OPEN] > 0.5
    walk = np.clip(ride[..., RIDE_FEAT_WALK], 0.0, None) * 3600.0
    wait = np.clip(ride[..., RIDE_FEAT_WAIT], 0.0, None) * 3600.0
    duration = np.clip(ride[..., RIDE_FEAT_DURATION], 0.0, None) * 900.0

    time_left_frac = np.clip(guest[..., GUEST_FEAT_TIME_LEFT], 0.0, None)
    remaining_sec = time_left_frac * DAY_SECONDS
    day_frac = env[..., 0]
    soft_closed = (day_frac >= 1.0) | (time_left_frac <= 0.0)

    drain = np.where(
        day_frac < 1.0,
        np.full((batch,), CLOSE_DRAIN_SEC, dtype=np.float32),
        np.zeros((batch,), dtype=np.float32),
    )
    remaining_for_feas = (remaining_sec + drain)[..., None]
    time_ok = (walk + wait + duration) <= remaining_for_feas

    at_ride_node = guest[..., GUEST_FEAT_AT_RIDE_NODE] > 0.5
    already_here = at_ride_node[..., None] & (ride[..., RIDE_FEAT_WALK] <= 1e-6)

    ride_ok = open_ok & time_ok & (~already_here) & (~soft_closed[..., None])

    mask = np.zeros((batch, NUM_ACTIONS), dtype=bool)
    mask[:, :num_rides] = ride_ok
    mask[:, NUM_RIDES] = True
    mask[:, NUM_RIDES + 1] = ~soft_closed
    return mask


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def _read_meta(path: Path) -> dict:
    meta_path = path.with_suffix(path.suffix + ".json")
    if not meta_path.is_file():
        # Also accept foo.onnx → foo.json (export writes this).
        alt = path.with_suffix(".json")
        meta_path = alt if alt.is_file() else meta_path
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


class Recommender:
    def __init__(
        self,
        checkpoint: Path | str | None = None,
        device: str | None = None,
        *,
        version: str | None = None,
    ) -> None:
        self.device = device or settings.DEVICE
        self.version = version
        explicit = checkpoint is not None
        if checkpoint is not None:
            path = Path(checkpoint)
        elif version is not None:
            path = Path(settings.MODELS[version])
        else:
            path = Path(settings.MODELS[settings.DEFAULT_MODEL_VERSION])
        self.checkpoint_path = path
        self._session = None
        self._torch_model = None
        self.step = 0
        self.meta: dict = {}
        self.is_stub = False
        self._backend = "none"
        self._load(path, allow_write_stub=explicit)
        label = f" [{version}]" if version else ""
        if self.is_stub:
            logger.warning(
                "Using STUB model%s (%s, step=%s). Place a real checkpoint at this path and restart.",
                label,
                self.checkpoint_path,
                self.step,
            )
        else:
            logger.info(
                "Loaded companion model%s %s via %s (step=%s)",
                label,
                self.checkpoint_path,
                self._backend,
                self.step,
            )

    def _load(self, path: Path, *, allow_write_stub: bool) -> None:
        if path.is_file() and path.suffix.lower() == ".onnx":
            self._load_onnx(path)
            return
        if path.is_file() and path.suffix.lower() == ".pt":
            self._load_torch(path)
            return
        # Prefer ONNX stub for deploy images without torch.
        stub_path = path if allow_write_stub else _STUB_ONNX
        if not allow_write_stub:
            logger.warning(
                "Checkpoint %s not found — loading disposable stub at %s.",
                path,
                stub_path,
            )
        self._load_or_create_stub(stub_path)

    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        meta = _read_meta(path)
        self.step = int(meta.get("step", 0))
        self.meta = {k: v for k, v in meta.items() if k != "step"}
        self.is_stub = bool(self.meta.get("stub"))
        self._backend = "onnxruntime"
        self.checkpoint_path = path

    def _load_torch(self, path: Path) -> None:
        import torch

        from Park.model import forward_with_mask, obs_flat_to_tensors
        from Park.training.checkpoint import load_checkpoint

        model, step, meta = load_checkpoint(path, self.device)
        model.eval()
        self._torch_model = model
        self._torch_forward = forward_with_mask
        self._torch_split = obs_flat_to_tensors
        self._torch = torch
        self.step = int(step)
        self.meta = dict(meta or {})
        self.is_stub = bool(self.meta.get("stub"))
        self._backend = "torch"
        self.checkpoint_path = path

    def _load_or_create_stub(self, path: Path) -> None:
        if path.is_file():
            if path.suffix.lower() == ".onnx":
                self._load_onnx(path)
                return
            self._load_torch(path)
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        # Stub creation needs torch (dev/tests only). Deploy images ship real .onnx files.
        try:
            import torch
            import torch.nn as nn

            from Park.training.checkpoint import default_model, save_checkpoint
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                f"Missing model {path} and cannot create stub (need torch): {exc}"
            ) from exc

        if path.suffix.lower() == ".onnx":
            model = default_model("cpu").eval()

            class _Wrap(nn.Module):
                def __init__(self, m: nn.Module) -> None:
                    super().__init__()
                    self.model = m

                def forward(self, guest, ride, env):
                    logits, _values = self.model(guest, ride, env)
                    return logits

            guest = torch.zeros(1, GUEST_FEAT_DIM)
            ride = torch.zeros(1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
            env = torch.zeros(1, ENV_DYNAMIC_FEAT_DIM)
            with torch.inference_mode():
                torch.onnx.export(
                    _Wrap(model),
                    (guest, ride, env),
                    str(path),
                    input_names=["guest", "ride", "env"],
                    output_names=["logits"],
                    opset_version=17,
                    dynamo=False,
                )
            path.with_suffix(".json").write_text(
                json.dumps({"step": 0, "stub": True, "path": str(path)}, indent=2),
                encoding="utf-8",
            )
            self._load_onnx(path)
            self.is_stub = True
            return

        model = default_model(self.device)
        save_checkpoint(path, model, optimizer=None, step=0, extra={"stub": True})
        self._load_torch(path)

    def info(self) -> dict:
        return {
            "version": self.version,
            "path": str(self.checkpoint_path),
            "step": int(self.step),
            "stub": self.is_stub,
            "device": str(self.device),
            "backend": self._backend,
            "available": self.checkpoint_path.is_file() and not self.is_stub,
        }

    def recommend(self, obs_flat: np.ndarray) -> dict:
        guest, ride, env = _split_flat_obs(obs_flat)
        legal = build_action_mask_numpy(guest, ride, env)[0]

        if self._session is not None:
            logits = self._session.run(
                None, {"guest": guest, "ride": ride, "env": env}
            )[0]
            logits = np.asarray(logits, dtype=np.float32)[0].copy()
            logits[~legal] = -1.0e9
        elif self._torch_model is not None:
            torch = self._torch
            with torch.no_grad():
                obs = torch.tensor(
                    np.asarray(obs_flat, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                g, r, e = self._torch_split(obs)
                masked, _, m = self._torch_forward(self._torch_model, g, r, e)
                logits = masked[0].cpu().numpy()
                legal = m[0].cpu().numpy().astype(bool)
        else:
            raise RuntimeError("No model backend loaded")

        probs = _softmax(logits)
        action = int(probs.argmax())

        distribution = []
        for i in range(NUM_ACTIONS):
            distribution.append(
                {
                    "action_id": i,
                    "label": action_label(i),
                    "prob": float(probs[i]),
                    "legal": bool(legal[i]),
                    "is_ride": i < len(ACTION_LABELS) - 2,
                }
            )
        distribution.sort(key=lambda row: row["prob"], reverse=True)

        return {
            "recommended": {
                "action_id": action,
                "label": action_label(action),
                "prob": float(probs[action]),
                "legal": bool(legal[action]),
            },
            "distribution": distribution,
            "model": self.info(),
        }


class ModelRegistry:
    """Lazy-loads configured model versions (keeps them warm once used)."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or settings.DEVICE
        self.default_version = settings.DEFAULT_MODEL_VERSION
        if self.default_version not in settings.MODELS:
            raise ValueError(
                f"DEFAULT_MODEL_VERSION={self.default_version!r} missing from MODELS"
            )
        self._by_version: dict[str, Recommender] = {}
        # Warm the default so /api/health and first recommend are ready.
        self.get(self.default_version)

    def versions(self) -> list[dict]:
        out = []
        for version, path in settings.MODELS.items():
            if version in self._by_version:
                info = self._by_version[version].info()
            else:
                p = Path(path)
                exists = p.is_file()
                info = {
                    "version": version,
                    "path": str(p),
                    "step": int(_read_meta(p).get("step", 0)) if exists else 0,
                    "stub": False if exists else True,
                    "device": str(self.device),
                    "backend": "onnxruntime" if p.suffix.lower() == ".onnx" else "torch",
                    "available": exists,
                }
            out.append({"id": version, "label": version.upper(), **info})
        return out

    def get(self, version: str | None = None) -> Recommender:
        key = version or self.default_version
        if key not in settings.MODELS:
            raise KeyError(key)
        if key not in self._by_version:
            self._by_version[key] = Recommender(device=self.device, version=key)
        return self._by_version[key]
