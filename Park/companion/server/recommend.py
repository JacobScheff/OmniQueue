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
    route_k as default_route_k,
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


def _adapt_ride_feat_dim(ride: np.ndarray, expected_dim: int) -> np.ndarray:
    """Slice or zero-pad ride feats so the last axis matches the loaded model."""
    cur = int(ride.shape[-1])
    exp = int(expected_dim)
    if cur == exp:
        return ride
    if cur > exp:
        return np.ascontiguousarray(ride[..., :exp])
    pad = np.zeros((*ride.shape[:-1], exp - cur), dtype=ride.dtype)
    return np.concatenate([ride, pad], axis=-1)


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


def _dist_rows(
    logits: np.ndarray,
    legal: np.ndarray,
    *,
    action_dim: int | None = None,
) -> list[dict]:
    """Softmax over legal actions; return sorted DistRow dicts."""
    logits = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
    legal = np.asarray(legal, dtype=bool).reshape(-1)
    n = int(action_dim or min(logits.shape[0], legal.shape[0]))
    logits = logits[:n]
    legal = legal[:n]
    masked = logits.copy()
    masked[~legal] = -1.0e9
    if not legal.any():
        probs = np.zeros(n, dtype=np.float32)
    else:
        probs = _softmax(masked)
    rows = []
    for i in range(n):
        rows.append(
            {
                "action_id": i,
                "label": action_label(i),
                "prob": float(probs[i]),
                "legal": bool(legal[i]),
                "is_ride": i < NUM_RIDES,
            }
        )
    rows.sort(key=lambda row: row["prob"], reverse=True)
    return rows


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
        self._onnx_has_route = False
        self._onnx_has_slots = False
        self._onnx_has_force = False
        self._onnx_has_force_any_slot = False
        self._onnx_outputs: list[str] = []
        self.ride_feat_dim = RIDE_DYNAMIC_FEAT_DIM
        self.route_k = default_route_k()
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
        self.route_k = int(self.meta.get("route_k", default_route_k()))
        outs = [o.name for o in self._session.get_outputs()]
        ins = {i.name: i for i in self._session.get_inputs()}
        self._onnx_outputs = outs
        self._onnx_has_route = "route" in outs and "slot0_logits" in outs
        self._onnx_has_slots = "slot_logits" in outs and "slot_masks" in outs
        self._onnx_has_force_any_slot = "force_slot" in ins and "force_action" in ins
        self._onnx_has_force = self._onnx_has_force_any_slot or "force_first" in ins
        ride_dim = self.meta.get("ride_dynamic_feat_dim")
        if ride_dim is None and "ride" in ins:
            shape = ins["ride"].shape
            if shape is not None and len(shape) >= 1:
                last = shape[-1]
                if isinstance(last, int) and last > 0:
                    ride_dim = last
        self.ride_feat_dim = int(ride_dim or RIDE_DYNAMIC_FEAT_DIM)

    def _load_torch(self, path: Path) -> None:
        import torch

        from Park.model import forward_route_with_mask
        from Park.training.checkpoint import load_checkpoint

        model, step, meta = load_checkpoint(path, self.device)
        model.eval()
        self._torch_model = model
        self._torch_forward_route = forward_route_with_mask
        self._torch = torch
        self.step = int(step)
        self.meta = dict(meta or {})
        self.is_stub = bool(self.meta.get("stub"))
        self._backend = "torch"
        self.checkpoint_path = path
        self.route_k = int(getattr(model, "route_k", default_route_k()))
        cfg_dim = None
        if isinstance(meta, dict):
            cfg_dim = meta.get("ride_dynamic_feat_dim")
        if cfg_dim is None:
            try:
                cfg_dim = int(model.ride_feat_proj[0].in_features)
            except Exception:  # noqa: BLE001
                cfg_dim = RIDE_DYNAMIC_FEAT_DIM
        self.ride_feat_dim = int(cfg_dim)

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

                def forward(self, guest, ride, env, force_slot, force_action):
                    out = self.model.forward_route(
                        guest,
                        ride,
                        env,
                        routes=None,
                        deterministic=True,
                        force_slot=force_slot,
                        force_action=force_action,
                    )
                    return (
                        out.routes.to(dtype=torch.int64),
                        out.slot0_logits,
                        out.slot_logits,
                        out.slot_masks.to(dtype=torch.float32),
                    )

            guest = torch.zeros(1, GUEST_FEAT_DIM)
            guest[..., GUEST_FEAT_TIME_LEFT] = 0.5
            guest[..., :NUM_RIDES] = 1.0 / float(NUM_RIDES)
            ride = torch.zeros(1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
            ride[..., RIDE_FEAT_OPEN] = 1.0
            ride[..., 5] = 0.1
            env = torch.zeros(1, ENV_DYNAMIC_FEAT_DIM)
            force_slot = torch.full((1,), -1, dtype=torch.int64)
            force_action = torch.full((1,), -1, dtype=torch.int64)
            with torch.inference_mode():
                torch.onnx.export(
                    _Wrap(model),
                    (guest, ride, env, force_slot, force_action),
                    str(path),
                    input_names=["guest", "ride", "env", "force_slot", "force_action"],
                    output_names=["route", "slot0_logits", "slot_logits", "slot_masks"],
                    opset_version=17,
                    dynamo=False,
                )
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "step": 0,
                        "stub": True,
                        "path": str(path),
                        "arch_version": "rank_route_v1",
                        "route_k": int(model.route_k),
                        "ride_dynamic_feat_dim": int(RIDE_DYNAMIC_FEAT_DIM),
                    },
                    indent=2,
                ),
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
            "supports_force_first": self._backend == "torch" or self._onnx_has_force,
            "supports_force_any_slot": (
                self._backend == "torch" or self._onnx_has_force_any_slot
            ),
            "supports_slot_distributions": (
                self._backend == "torch" or self._onnx_has_slots
            ),
            "route_k": int(self.route_k),
            "ride_dynamic_feat_dim": int(self.ride_feat_dim),
        }

    def _force_feeds(self, slot: int | None, action: int | None) -> dict[str, np.ndarray]:
        """Build ONNX force inputs, preferring force_slot/force_action, falling
        back to the legacy single force_first input (slot 0 only)."""
        if self._onnx_has_force_any_slot:
            return {
                "force_slot": np.asarray(
                    [slot if slot is not None else -1], dtype=np.int64
                ),
                "force_action": np.asarray(
                    [action if action is not None else -1], dtype=np.int64
                ),
            }
        if self._onnx_has_force:
            ff = action if (slot == 0 and action is not None) else -1
            return {"force_first": np.asarray([ff], dtype=np.int64)}
        return {}

    def recommend(
        self,
        obs_flat: np.ndarray,
        *,
        force_slot: int | None = None,
        force_action: int | None = None,
    ) -> dict:
        guest, ride, env = _split_flat_obs(obs_flat)
        # Masking uses the full live ride tensor; models may expect a legacy width.
        model_ride = _adapt_ride_feat_dim(ride, self.ride_feat_dim)
        legal = build_action_mask_numpy(guest, ride, env)[0]
        route_ids: list[int] = []
        slot_logits_np: np.ndarray | None = None
        slot_masks_np: np.ndarray | None = None

        if force_slot is not None or force_action is not None:
            if force_slot is None or force_action is None:
                raise ValueError("force_slot and force_action must be set together")
            force_slot = int(force_slot)
            force_action = int(force_action)
            if force_slot < 0 or force_slot >= self.route_k:
                raise ValueError(
                    f"force_slot must be in [0, {self.route_k}), got {force_slot}"
                )
            if force_action < 0 or force_action >= NUM_ACTIONS:
                raise ValueError(
                    f"force_action must be in [0, {NUM_ACTIONS}), got {force_action}"
                )
            if force_slot > 0 and force_action >= NUM_RIDES:
                raise ValueError(
                    "force_action must be a ride (not exit/idle) for slots after the first"
                )
            if force_slot == 0 and not bool(legal[force_action]):
                raise ValueError(
                    f"force_action {force_action} ({action_label(force_action)}) "
                    "is not legal under the current mask"
                )
            if force_slot > 0 and not self._onnx_has_force_any_slot and self._session is not None:
                raise RuntimeError(
                    "This model can only force route slot 0; re-export with "
                    "Park/tools/export_companion_onnx.py to force later stops."
                )
        else:
            force_slot = None
            force_action = None

        if self._session is not None:
            feeds: dict[str, np.ndarray] = {
                "guest": guest,
                "ride": model_ride,
                "env": env,
            }
            if force_slot is not None and not self._onnx_has_force:
                raise RuntimeError(
                    "This ONNX model does not support force pins; re-export with "
                    "Park/tools/export_companion_onnx.py (arch rank_route_v1)."
                )
            feeds.update(self._force_feeds(force_slot, force_action))

            if getattr(self, "_onnx_has_route", False):
                if self._onnx_has_slots:
                    route_out, logits, slot_logits_np, slot_masks_np = self._session.run(
                        ["route", "slot0_logits", "slot_logits", "slot_masks"],
                        feeds,
                    )
                    slot_logits_np = np.asarray(slot_logits_np, dtype=np.float32)
                    slot_masks_np = np.asarray(slot_masks_np, dtype=np.float32) > 0.5
                else:
                    route_out, logits = self._session.run(
                        ["route", "slot0_logits"], feeds
                    )
                route_ids = [
                    int(x)
                    for x in np.asarray(route_out).reshape(-1).tolist()
                    if int(x) >= 0
                ]
                logits = np.asarray(logits, dtype=np.float32)[0].copy()
            else:
                logits = self._session.run(None, feeds)[0]
                logits = np.asarray(logits, dtype=np.float32)[0].copy()
            logits[~legal] = -1.0e9
        elif self._torch_model is not None:
            torch = self._torch
            with torch.no_grad():
                g = torch.tensor(guest, dtype=torch.float32, device=self.device)
                r = torch.tensor(model_ride, dtype=torch.float32, device=self.device)
                e = torch.tensor(env, dtype=torch.float32, device=self.device)
                fs = None
                fa = None
                if force_slot is not None:
                    fs = torch.tensor(
                        [force_slot], dtype=torch.long, device=self.device
                    )
                    fa = torch.tensor(
                        [force_action], dtype=torch.long, device=self.device
                    )
                out = self._torch_forward_route(
                    self._torch_model,
                    g,
                    r,
                    e,
                    routes=None,
                    deterministic=True,
                    force_slot=fs,
                    force_action=fa,
                    # Always fully deterministic: any two requests with the same
                    # inputs must agree on the plan, so a route slot shown to the
                    # guest is guaranteed still reachable if they force-pick it
                    # (or a later slot) in a follow-up request.
                    close_margin=0.0,
                )
                logits = out.slot0_logits[0].cpu().numpy()
                legal = out.slot0_mask[0].cpu().numpy().astype(bool)
                route_ids = [
                    int(x)
                    for x in out.routes[0].cpu().numpy().tolist()
                    if int(x) >= 0
                ]
                slot_logits_np = out.slot_logits.cpu().numpy()
                slot_masks_np = out.slot_masks.cpu().numpy().astype(bool)
        else:
            raise RuntimeError("No model backend loaded")

        probs = _softmax(logits)
        natural_action = int(probs.argmax())
        if not route_ids:
            forced_slot0 = force_action if (force_slot == 0 and force_action is not None) else None
            route_ids = [forced_slot0 if forced_slot0 is not None else natural_action]

        # Recommended follows the plan's committed next action (forced when set).
        action = int(route_ids[0])
        if force_slot is not None:
            if force_slot >= len(route_ids) or route_ids[force_slot] != force_action:
                raise RuntimeError(
                    f"Model did not honor force at slot {force_slot} "
                    f"action={force_action} ({action_label(force_action)}); it may not "
                    "be reachable as a candidate at that stop, or the export is stale."
                )

        distribution = _dist_rows(logits, legal, action_dim=NUM_ACTIONS)

        distributions_by_slot: list[list[dict]] = []
        if slot_logits_np is not None and slot_masks_np is not None:
            # Shapes: (1, K, A) or (K, A)
            sl = np.asarray(slot_logits_np)
            sm = np.asarray(slot_masks_np)
            if sl.ndim == 3:
                sl = sl[0]
                sm = sm[0]
            n_slots = min(len(route_ids), sl.shape[0])
            for k in range(n_slots):
                # Slot 0: full action dim; later slots: rides only (exit/idle never legal).
                dim = NUM_ACTIONS if k == 0 else NUM_RIDES
                distributions_by_slot.append(
                    _dist_rows(sl[k, :dim], sm[k, :dim], action_dim=dim)
                )
        else:
            distributions_by_slot = [distribution]

        # Keep top-level distribution as slot 0 for backward compatibility.
        distribution = distributions_by_slot[0]

        route = [
            {
                "action_id": int(aid),
                "label": action_label(int(aid)),
                "slot": slot,
                "is_ride": int(aid) < NUM_RIDES,
                "prob_slot": next(
                    (
                        float(row["prob"])
                        for row in distributions_by_slot[slot]
                        if row["action_id"] == int(aid)
                    ),
                    None,
                )
                if slot < len(distributions_by_slot)
                else None,
            }
            for slot, aid in enumerate(route_ids)
        ]

        return {
            "recommended": {
                "action_id": action,
                "label": action_label(action),
                "prob": float(probs[action]) if action < len(probs) else 0.0,
                "legal": bool(legal[action]) if action < len(legal) else False,
            },
            "natural_recommended": {
                "action_id": natural_action,
                "label": action_label(natural_action),
                "prob": float(probs[natural_action]),
                "legal": bool(legal[natural_action]),
            },
            "forced_slot": force_slot,
            "forced_action": force_action,
            "route": route,
            "distribution": distribution,
            "distributions_by_slot": distributions_by_slot,
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
                    "supports_force_first": False,
                    "supports_force_any_slot": False,
                    "supports_slot_distributions": False,
                    "route_k": int(_read_meta(p).get("route_k", default_route_k()))
                    if exists
                    else default_route_k(),
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
