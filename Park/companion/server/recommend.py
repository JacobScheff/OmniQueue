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

# Dedicated path for auto-generated random weights — never overwrite MODEL_PATH.
_STUB_PATH = Path(__file__).resolve().parents[1] / "model" / "_stub_random.pt"


class Recommender:
    def __init__(self, checkpoint: Path | str | None = None, device: str | None = None) -> None:
        self.device = torch.device(device or settings.DEVICE)
        explicit = checkpoint is not None
        path = Path(checkpoint) if explicit else Path(settings.MODEL_PATH)
        self.checkpoint_path = path
        self.model, self.step, self.meta = self._load(path, allow_write_stub=explicit)
        self.model.eval()
        self.is_stub = bool(self.meta.get("stub"))
        if self.is_stub:
            logger.warning(
                "Using STUB model (%s, step=%s). Replace companion/settings.py MODEL_PATH "
                "with a real training checkpoint and restart the server.",
                self.checkpoint_path,
                self.step,
            )
        else:
            logger.info(
                "Loaded PPO checkpoint %s (step=%s)",
                self.checkpoint_path,
                self.step,
            )

    def _load(self, path: Path, *, allow_write_stub: bool):
        if path.is_file():
            model, step, meta = load_checkpoint(path, self.device)
            if meta.get("stub"):
                logger.warning(
                    "%s is a stub checkpoint (extra.stub=true, step=%s). "
                    "Delete it and copy your real .pt to MODEL_PATH, then restart.",
                    path,
                    step,
                )
            return model, step, meta

        # Never write a stub over the configured MODEL_PATH — that previously
        # left users with a fake file named like their real checkpoint.
        stub_path = path if allow_write_stub else _STUB_PATH
        if not allow_write_stub:
            logger.warning(
                "MODEL_PATH %s not found — loading disposable stub at %s. "
                "Copy your trained .pt to MODEL_PATH and restart.",
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
