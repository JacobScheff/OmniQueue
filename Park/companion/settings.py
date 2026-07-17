"""Companion runtime settings — edit this file instead of using env vars.

Restart the server after changing anything here (settings are read at startup).
"""

from __future__ import annotations

from pathlib import Path

# Trained PPO checkpoint (.pt). Point this at your weights.
MODEL_PATH = Path(__file__).resolve().parent / "model" / "ppo_step_2543143.pt"

# Torch device for inference ("cpu" or "cuda").
DEVICE = "cuda"

# How long to cache ThemeParks.wiki live waits (seconds).
WAIT_CACHE_TTL_SEC = 45.0

# Bind address for the API / static server.
HOST = "0.0.0.0"
PORT = 8000

# Set True only for local frontend iteration (uvicorn --reload).
RELOAD = False
