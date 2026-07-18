"""Load PPO checkpoints and run single-party live inference."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from Park.companion import settings
from Park.companion.server.obs import ACTION_LABELS, action_label
from Park.model import forward_with_mask, obs_flat_to_tensors
from Park.training.checkpoint import default_model, load_checkpoint, save_checkpoint
from Park.training.features import NUM_ACTIONS

logger = logging.getLogger(__name__)

# Dedicated path for auto-generated random weights — never overwrite configured models.
_STUB_PATH = Path(__file__).resolve().parents[1] / "model" / "_stub_random.pt"


class Recommender:
    def __init__(
        self,
        checkpoint: Path | str | None = None,
        device: str | None = None,
        *,
        version: str | None = None,
    ) -> None:
        self.device = torch.device(device or settings.DEVICE)
        self.version = version
        explicit = checkpoint is not None
        if checkpoint is not None:
            path = Path(checkpoint)
        elif version is not None:
            path = Path(settings.MODELS[version])
        else:
            path = Path(settings.MODELS[settings.DEFAULT_MODEL_VERSION])
        self.checkpoint_path = path
        self.model, self.step, self.meta = self._load(path, allow_write_stub=explicit)
        self.model.eval()
        self.is_stub = bool(self.meta.get("stub"))
        label = f" [{version}]" if version else ""
        if self.is_stub:
            logger.warning(
                "Using STUB model%s (%s, step=%s). Place a real .pt at this path and restart.",
                label,
                self.checkpoint_path,
                self.step,
            )
        else:
            logger.info(
                "Loaded PPO checkpoint%s %s (step=%s)",
                label,
                self.checkpoint_path,
                self.step,
            )

    def _load(self, path: Path, *, allow_write_stub: bool):
        if path.is_file():
            model, step, meta = load_checkpoint(path, self.device)
            if meta.get("stub"):
                logger.warning(
                    "%s is a stub checkpoint (extra.stub=true, step=%s).",
                    path,
                    step,
                )
            return model, step, meta

        stub_path = path if allow_write_stub else _STUB_PATH
        if not allow_write_stub:
            logger.warning(
                "Checkpoint %s not found — loading disposable stub at %s.",
                path,
                stub_path,
            )
        return self._load_or_create_stub(stub_path)

    def _load_or_create_stub(self, path: Path):
        if path.is_file():
            return load_checkpoint(path, self.device)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = default_model(self.device)
        save_checkpoint(path, model, optimizer=None, step=0, extra={"stub": True})
        return model, 0, {"stub": True}

    def info(self) -> dict:
        return {
            "version": self.version,
            "path": str(self.checkpoint_path),
            "step": int(self.step),
            "stub": self.is_stub,
            "device": str(self.device),
            "available": self.checkpoint_path.is_file() and not self.is_stub,
        }

    @torch.no_grad()
    def recommend(self, obs_flat: np.ndarray) -> dict:
        obs = torch.tensor(
            np.asarray(obs_flat, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        guest, ride, env = obs_flat_to_tensors(obs)
        logits, _, mask = forward_with_mask(self.model, guest, ride, env)
        probs = torch.softmax(logits[0, 0], dim=-1).cpu().numpy()
        legal = mask[0, 0].cpu().numpy().astype(bool)
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
    """Keeps every configured model version warm in memory."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or settings.DEVICE
        self.default_version = settings.DEFAULT_MODEL_VERSION
        if self.default_version not in settings.MODELS:
            raise ValueError(
                f"DEFAULT_MODEL_VERSION={self.default_version!r} missing from MODELS"
            )
        self._by_version: dict[str, Recommender] = {}
        for version in settings.MODELS:
            self._by_version[version] = Recommender(
                device=self.device,
                version=version,
            )

    def versions(self) -> list[dict]:
        return [
            {
                "id": version,
                "label": version.upper(),
                **self._by_version[version].info(),
            }
            for version in settings.MODELS
        ]

    def get(self, version: str | None = None) -> Recommender:
        key = version or self.default_version
        if key not in self._by_version:
            raise KeyError(key)
        return self._by_version[key]
