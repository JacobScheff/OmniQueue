# Live park companion

**Modules:** `companion/`, `companion/server/`, `companion/web/`

## Overview

Phone-first **live Disneyland companion**: pull real wait times from ThemeParks.wiki, let the guest edit preferences / must-dos / completions / location (with undo/redo + `localStorage`), and run the trained **PPO** checkpoint once (`G=1`) to show the next action and full masked probability distribution.

This does **not** run the C++ DES or Watch pygame UI. The simulator is unused at request time; only `ParkRouterModel`, walk times from `park_graph`, and `config.RIDES` are reused on the server.

## Run (dev)

```bash
# from repo root (parent of Park/) so `import Park...` resolves
cd /path/to/workspace
pip install -r Park/companion/requirements.txt
pip install -e Park   # or existing omniqueue editable install

# API (creates a random stub checkpoint on first run if none is provided)
COMPANION_MODEL_PATH=checkpoints/ppo/ppo_final.pt \
  python -m Park.companion.server

# Frontend (separate terminal)
cd Park/companion/web && npm install && npm run dev
```

Open `http://127.0.0.1:5173` (Vite proxies `/api` → `:8000`).

## Run (production-ish)

```bash
cd Park/companion/web && npm ci && npm run build
COMPANION_MODEL_PATH=/models/ppo_final.pt python -m Park.companion.server
# serves API + web/dist on :8000
```

Docker:

```bash
docker build -f Park/companion/Dockerfile -t omniqueue-companion .
docker run --rm -p 8000:8000 \
  -e COMPANION_MODEL_PATH=/app/model/ppo_final.pt \
  -v /path/to/ppo_final.pt:/app/model/ppo_final.pt:ro \
  omniqueue-companion
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Model + wait cache status |
| GET | `/api/catalog` | Ride/hub list + default prefs |
| GET | `/api/waits` | Cached live board (`?force=true`) |
| POST | `/api/recommend` | Body: prefs, must-dos, history, location → distribution |

## Observation mapping

Live features are built in `companion/server/obs.py` to match training (`FLAT_OBS_DIM=322`):

| Slot | Source |
|------|--------|
| Wait / open | ThemeParks.wiki standby + status (`DOWN`/`CLOSED` → closed) |
| Duration / capacity | `config.RIDES` |
| Walk | `park_graph.walk_times_to_rides` from user location |
| History / must-do | User inputs |
| Prefs | User weights (L1-normalized) |
| Incoming | **0** (not available from the public API) |
| Env mean wait / broken frac | Aggregated from the live board |

## Persistence

All guest state (prefs, must-dos, completions, location, leave time, undo/redo stacks) lives in the browser `localStorage` key `omniqueue-companion-v1`. The API is stateless per request.

## Env vars

| Var | Default | Meaning |
|-----|---------|---------|
| `COMPANION_MODEL_PATH` | auto-discover / stub | PPO `.pt` checkpoint |
| `COMPANION_DEVICE` | `cpu` | Torch device |
| `COMPANION_WAIT_TTL` | `45` | Seconds to cache live waits |
| `COMPANION_HOST` / `COMPANION_PORT` | `0.0.0.0` / `8000` | Bind address |

## Notes

- Credit [ThemeParks.wiki](https://themeparks.wiki) for wait data; respect their rate guidance (companion caches ~45s).
- A **stub** random checkpoint is created under `companion/model/ppo_live.pt` when no trained weights are found so the UI can be exercised offline. Replace it with your PPO run before trusting recommendations.
