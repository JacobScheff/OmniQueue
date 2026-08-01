# Live park companion

**Modules:** `companion/`, `companion/server/`, `companion/web/`, `run_companion.py`

## Overview

Phone-first **live Disneyland companion**: pull real wait times from ThemeParks.wiki, let the guest edit preferences / must-dos / completions / location (with undo/redo + `localStorage`), and run the trained **PPO** checkpoint once (single-party / no guest axis) to show the **committed next action**, a short **planned route** (`route[0..K-1]`), and **per-slot** masked probability distributions. Guests can **force a legal first ride** (`force_first`) so the autoregressive decoder continues the rest of the plan from that pin.

This does **not** run the C++ DES or Watch pygame UI. The simulator is unused at request time; only the exported ONNX policy, walk times from `park_graph`, and `config.RIDES` are reused on the server.

## Configure

Edit `companion/settings.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `MODELS` | `v1`/`v2` → `v1.onnx`/`v2.onnx` | Named checkpoints (UI tags; add keys for more versions) |
| `DEFAULT_MODEL_VERSION` | `"v1"` | Tag used when the client omits one |
| `DEVICE` / `COMPANION_DEVICE` | `"cpu"` | Device hint (ONNX Runtime uses CPU in deploy) |
| `WAIT_CACHE_TTL_SEC` | `45` | Live-wait cache lifetime |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address (`PORT` overridden by hosts) |
| `RELOAD` | `False` | Uvicorn auto-reload (dev only) |

Place real `.onnx` files under `companion/model/` for each tag (committed for deploy). Missing files fall back to a disposable stub when torch is installed (banner warns you). Version tags in the UI come from `MODELS` keys (**V1**, **V2**); `/api/recommend` accepts `model_version`.

### Refreshing ONNX after training

You do **not** need this for an ordinary deploy — ONNX files are already in the repo. Only re-run when a `.pt` checkpoint changes:

```bash
pip install torch onnx onnxruntime
PYTHONPATH=. python Park/tools/export_companion_onnx.py
```

Route models export ONNX (`arch_version=route_v2`) with inputs `guest`, `ride`, `env`, `force_first` (int64, `-1` = no pin) and outputs `route` (int64, length K), `slot0_logits`, `slot_logits` `(1,K,A)`, `slot_masks` `(1,K,A)` float 0/1. Older `route`+`slot0_logits` ONNX still loads (slot-0 distribution only; `force_first` requests fail until re-export). Single-`logits` ONNX still loads as a length-1 route from argmax. Export uses a mid-day open-park example (non-zero `time_left`) so the autoregressive ride-update path is traced; the smoke check requires Torch/ORT route equality, no repeated rides, and a working `force_first` pin.

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

Docker (single process: API + built SPA + ONNX):

```bash
docker build -f Park/companion/Dockerfile -t omniqueue-companion .
docker run --rm -p 8000:8000 omniqueue-companion
```

## Free hosting (one deployment)

**Recommended: [Render](https://render.com) free web service** (SPA + API together; sleeps after ~15 min idle).

1. Push this branch / merge to the repo Render will watch.
2. In Render: **New → Blueprint** and select the repo (uses root `render.yaml`), **or** **New → Web Service**:
   - Runtime: **Docker**
   - Dockerfile path: `Park/companion/Dockerfile`
   - Docker context: repo root (`.`)
   - Instance: **Free**
   - Health check path: `/api/health`
3. Deploy. Open the `*.onrender.com` URL — the phone UI and `/api/*` are on the same origin.

Cold start after sleep can take ~30–60s while the free instance wakes. No ONNX rebuild is required on your machine for deploy.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Model + wait cache status |
| GET | `/api/catalog` | Ride/hub list + default prefs |
| GET | `/api/waits` | Cached live board (`?force=true`) |
| POST | `/api/recommend` | Body: prefs, must-dos, history, location, optional `force_first` → `recommended`, `route`, `distribution` (slot 0), `distributions_by_slot`, `natural_recommended`, `forced_first` |

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

## What-if force-first + slot distributions

- Request field `force_first` (action id) pins route slot 0 when legal; the decoder GRU then continues slots `1..K-1` under the tail mask (open ∧ unfinished ∧ not already picked). Slot-0 logits stay the natural policy — only the chosen action changes.
- Response includes `distributions_by_slot[k]` (softmax over that step's mask), `natural_recommended` (slot-0 argmax without the pin), and `forced_first`.
- UI: tap a legal ride in the **Next** distribution to pin/unpin; slot tabs switch which step's distribution is shown. Force state is ephemeral (not in `localStorage`).

## Persistence

All guest state (prefs, must-dos, completions, location, leave time, undo/redo stacks) lives in the browser `localStorage` key `omniqueue-companion-v1`. The API is stateless per request. Force-first exploration is UI-only and is cleared when switching model versions.

## Notes

- Credit [ThemeParks.wiki](https://themeparks.wiki) for wait data; respect their rate guidance (companion caches ~45s).
- `run_companion.py` inserts the parent of `Park/` onto `sys.path` so you do not need to set `PYTHONPATH` (the editable wheel only installs `_park_sim`, not the Python package tree).
- Deploy image uses **onnxruntime** only (no PyTorch) so it fits Render’s free ~512 MB RAM tier.
- `Park.training.features` keeps dimension constants importable without torch; torch masking helpers load torch lazily for training.
- Walk times load from committed `data/walk_matrix.npz` (do not rely on gitignored `cache/` in deploy). Rebuild with `python -c "from Park.park_graph import get_park_graph, reset_park_graph; reset_park_graph(); get_park_graph(force_recompute=True)"` then copy `cache/walk_matrix.npz` → `data/walk_matrix.npz` if pathways/config change.
