"""Companion runtime settings — edit this file instead of using env vars.

Restart the server after changing anything here (settings are read at startup).
"""

from __future__ import annotations

from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent / "model"

# Named PPO checkpoints. Add v3, v4, … the same way.
# Tags shown in the UI come from these keys (v1, v2, …).
MODELS: dict[str, Path] = {
    "v1": _MODEL_DIR / "ppo_step_2543143.pt",
    "v2": _MODEL_DIR / "ppo_v2.pt",
}

# Default tag when the client does not specify one.
DEFAULT_MODEL_VERSION = "v2"

# Torch device for inference ("cpu" or "cuda").
DEVICE = "cuda"

# How long to cache ThemeParks.wiki live waits (seconds).
WAIT_CACHE_TTL_SEC = 45.0

# Bind address for the API / static server.
HOST = "0.0.0.0"
PORT = 8000

# Set True only for local frontend iteration (uvicorn --reload).
RELOAD = False
