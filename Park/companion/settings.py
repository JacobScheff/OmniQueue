"""Companion runtime settings — edit this file instead of using env vars.

Restart the server after changing anything here (settings are read at startup).

Environment overrides (for free hosts like Render):
  PORT, HOST, COMPANION_DEVICE
"""

from __future__ import annotations

import os
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent / "model"

# Named PPO checkpoints (ONNX for free CPU hosting; .pt kept as training sources).
# Tags shown in the UI come from these keys (v1, v2, …). Add entries to restore multi-version.
# v1/v2: legacy ride feat dim 8 (no per-ride pref column). Recommend slices the live
# dim-9 obs before ONNX. v3: ride feat dim 9 with unfinished sharpened pref — place
# companion/model/v3.pt then run tools/export_companion_onnx.py --only v3.
MODELS: dict[str, Path] = {
    "v1": _MODEL_DIR / "v1.onnx",
    "v2": _MODEL_DIR / "v2.onnx",
    "v3": _MODEL_DIR / "v3.onnx",
}

# Default tag when the client does not specify one.
DEFAULT_MODEL_VERSION = "v1"

# Inference device hint ("cpu" or "cuda"). Free hosts are CPU-only.
# ONNX Runtime always uses CPUExecutionProvider in the companion image.
DEVICE = os.environ.get("COMPANION_DEVICE", "cpu")

# How long to cache ThemeParks.wiki live waits (seconds).
WAIT_CACHE_TTL_SEC = 45.0

# Bind address for the API / static server.
# PORT is overridden by most hosts (Render sets $PORT).
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Set True only for local frontend iteration (uvicorn --reload).
RELOAD = False
