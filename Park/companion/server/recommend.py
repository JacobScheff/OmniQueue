"""Load PPO checkpoint and run single-party live inference."""

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


class Recommender:
    def __init__(self, checkpoint: Path | str | None = None, device: str | None = None) -> None:
        self.device = torch.device(device or settings.DEVICE)
        path = self._resolve_checkpoint(checkpoint)
        self.checkpoint_path = path
        self.model, self.step, self.meta = self._load_or_create(path)
        self.model.eval()
        self.is_stub = bool(self.meta.get("stub"))

    @staticmethod
    def _resolve_checkpoint(explicit: Path | str | None) -> Path:
        if explicit is not None:
            return Path(explicit)
        configured = Path(settings.MODEL_PATH)
        if configured.is_file():
            return configured
        # Fall back to configured path (stub will be created there)
        return configured

    def _load_or_create(self, path: Path):
        if path.is_file():
            logger.info("Loading companion model from %s", path)
            return load_checkpoint(path, self.device)
        logger.warning(
            "No PPO checkpoint at %s — creating random stub weights. "
            "Set companion.settings.MODEL_PATH to your trained .pt file.",
            path,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        model = default_model(self.device)
        save_checkpoint(path, model, optimizer=None, step=0, extra={"stub": True})
        return model, 0, {"stub": True}

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
            "model": {
                "path": str(self.checkpoint_path),
                "step": int(self.step),
                "stub": self.is_stub,
                "device": str(self.device),
            },
        }
