# Live park companion

**Modules:** `companion/`, `companion/server/`, `companion/web/`, `run_companion.py`

## Overview

Phone-first **live Disneyland companion**: pull real wait times from ThemeParks.wiki, let the guest edit preferences / must-dos / completions / location (with undo/redo + `localStorage`), and run the trained **PPO** checkpoint once (`G=1`) to show the next action and full masked probability distribution.

This does **not** run the C++ DES or Watch pygame UI. The simulator is unused at request time; only `ParkRouterModel`, walk times from `park_graph`, and `config.RIDES` are reused on the server.

## Configure

Edit `companion/settings.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `MODELS` | `v1` → `ppo_step_2543143.pt`, `v2` → `ppo_v2.pt` | Named checkpoints (UI tags) |
| `DEFAULT_MODEL_VERSION` | `"v2"` | Tag used when the client omits one |
| `DEVICE` | `"cpu"` | Torch device |
| `WAIT_CACHE_TTL_SEC` | `45` | Live-wait cache lifetime |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address |
| `RELOAD` | `False` | Uvicorn auto-reload (dev only) |

Place real `.pt` files under `companion/model/` for each tag. Missing files fall back to a disposable stub (banner warns you). Switch versions in the UI with the **V1 / V2** tags; `/api/recommend` accepts `model_version`.

## Run (dev)

```bash
pip install -r Park/companion/requirements.txt

# API — run_companion.py makes `import Park` work without PYTHONPATH
python Park/run_companion.py

# Frontend (separate terminal)
cd Park/companion/web && npm install && npm run dev
```

Open `http://127.0.0.1:5173` (Vite proxies `/api` → `:8000`).

## Run (production-ish)

```bash
cd Park/companion/web && npm ci && npm run build
python Park/run_companion.py
# serves API + web/dist on :8000
```

Docker:

```bash
docker build -f Park/companion/Dockerfile -t omniqueue-companion .
docker run --rm -p 8000:8000 \
  -v /path/to/ppo_final.pt:/app/Park/companion/model/ppo_live.pt:ro \
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

## Notes

- Credit [ThemeParks.wiki](https://themeparks.wiki) for wait data; respect their rate guidance (companion caches ~45s).
- `run_companion.py` inserts the parent of `Park/` onto `sys.path` so you do not need to set `PYTHONPATH` (the editable wheel only installs `_park_sim`, not the Python package tree).
