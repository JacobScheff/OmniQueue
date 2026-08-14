# Live park companion

**Modules:** `companion/`, `companion/server/`, `companion/web/`, `run_companion.py`

## Overview

Phone-first **live Disneyland companion**: pull real wait times from ThemeParks.wiki, let the guest edit preferences / must-dos / completions / location (with undo/redo + `localStorage`), and run the trained **PPO** checkpoint once (single-party / no guest axis) to show the **committed next action**, a short **planned route** (`route[0..K-1]`), and **per-slot** masked probability distributions. Guests can **pin any legal stop in the route** (`force_slot` + `force_action`, not just the first) so the autoregressive decoder continues the rest of the plan from that pin while every other slot keeps deciding naturally.

A native iOS client lives in `Companion app/OmniQueueCompanion.xcodeproj`. It bundles `v2.onnx`, builds the same observation on-device, and calls ThemeParks.wiki directly. It does not use GPS and does not expose the web model switcher. Plan → **Refresh waits** spins the clockwise glyph for the whole fetch (and at least one turn): iOS 18 uses `symbolEffect(.rotate)`; earlier OS versions drive angle from a `TimelineView` with animations stripped so parent transactions cannot freeze the rotation. App Store metadata URLs (marketing / privacy / support) are `https://www.jacobscheff.com/OmniQueue` (with `#privacy` / `#support` anchors); the Me tab and disclaimer link the same pages. See `Companion app/README.md` for Connect submission notes. Paste `Companion app/app_review_notes.txt` into App Review Information → Notes on every submission; the Guideline 2.1 recording script is `Companion app/APP_REVIEW_INFORMATION.md`. Archive builds run `Companion app/scripts/patch_embedded_framework_plists.sh` so the copied `onnxruntime.framework` Info.plist has `MinimumOSVersion` (empty in the SPM binary, which Apple rejects as ITMS-90208).

This does **not** run the C++ DES or Watch pygame UI. The simulator is unused at request time; only the exported ONNX policy, walk times from `park_graph`, and `config.RIDES` are reused on the server.

## Configure

Edit `companion/settings.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `MODELS` | `v1`/`v2` → `v1.onnx`/`v2.onnx` | Named checkpoints (UI tags; add keys for more versions) |
| `DEFAULT_MODEL_VERSION` | `"v2"` | Tag used when the client omits one |
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

Route models export ONNX (`arch_version=rank_route_v1`) with inputs `guest`, `ride`, `env`, `force_slot` + `force_action` (both int64, `-1` = no pin) and outputs `route` (int64, length K), `slot0_logits`, `slot_logits` `(1,K,A)`, `slot_masks` `(1,K,A)` float 0/1. Older exports with a single `force_first` input still load, but only support pinning slot 0 (`Recommender.info().supports_force_any_slot` is `False` for those — re-export to pin later stops). Older `route`+`slot0_logits` ONNX still loads (slot-0 distribution only; force requests fail until re-export). Single-`logits` ONNX still loads as a length-1 route from argmax. Export uses a mid-day open-park example (non-zero `time_left`) so the autoregressive ride-update path is traced; the smoke check requires Torch/ORT route equality, no repeated rides, a working slot-0 force, and a working tail-slot force.

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

The default public URL is `https://<service-name>.onrender.com`. Root `render.yaml` sets `name: omniqueue`, so a Blueprint deploy should land at **`https://omniqueue.onrender.com`** (if that subdomain is free on Render).

### Fresh deploy (preferred)

1. Push / merge the branch Render will watch (include the `name: omniqueue` change in `render.yaml`).
2. In the Render dashboard, **delete** any old suspended service named `omniqueue-companion` (suspend alone keeps the old name reserved; you do not need that subdomain anymore).
3. **New → Blueprint** → connect the repo (uses root `render.yaml`). Confirm the web service is named **`omniqueue`**, then apply/deploy.
4. Or **New → Web Service** manually:
   - Name: **`omniqueue`** (this becomes the subdomain)
   - Runtime: **Docker**
   - Dockerfile path: `Park/companion/Dockerfile`
   - Docker context: repo root (`.`)
   - Instance: **Free**
   - Health check path: `/api/health`
   - Env: `COMPANION_DEVICE=cpu` (optional; Dockerfile already defaults to CPU)
5. Open **`https://omniqueue.onrender.com`** — the phone UI and `/api/*` are on the same origin. No app config change is needed for the new hostname.

If `omniqueue` is already taken on Render’s shared `*.onrender.com` namespace, pick another free name (or attach a custom domain later).

Cold start after sleep can take ~30–60s while the free instance wakes. No ONNX rebuild is required on your machine for deploy.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Model + wait cache status |
| GET | `/api/catalog` | Ride/hub list + default prefs |
| GET | `/api/waits` | Cached live board (`?force=true`) |
| POST | `/api/recommend` | Body: prefs, must-dos, history, location, optional `force_slot`+`force_action` → `recommended`, `route`, `distribution` (slot 0), `distributions_by_slot`, `natural_recommended`, `forced_slot`, `forced_action` |

## Observation mapping

Live features are built in `companion/server/obs.py` to match training (`FLAT_OBS_DIM=420`, `rank_route_v1`):

| Slot | Source |
|------|--------|
| Wait / open | ThemeParks.wiki standby + status (`DOWN`/`CLOSED` → closed) |
| Duration / capacity | `config.RIDES` |
| Walk | `park_graph.walk_times_to_rides` from user location |
| History / must-do | User inputs |
| Unfinished pref (ride feat 8) | Sharpened user pref when history is 0; else 0 |
| Prefs | User weights (L1-normalized) on guest feats |
| Incoming | **0** (not available from the public API) |
| Env mean wait / broken frac | Aggregated from the live board |

**Model versions:** `v1` / `v2` keep legacy ONNX with ride feat dim 8 — `recommend.py` slices off column 8 before inference. `v3` expects dim 9 (place `companion/model/v3.pt`, export with `tools/export_companion_onnx.py --only v3`).

Newer torch/ONNX graphs also refresh inter-ride walk features along the decoded path inside `forward_route` (no extra API inputs). Re-export after pulling decoder changes.

## What-if force-pick + slot distributions

- Request fields `force_slot` (0..`route_k`-1) + `force_action` pin **one** stop in the route to a specific legal action; every other slot — earlier or later — keeps deciding autoregressively/naturally. Slot 0 may be any legal action (including exit/idle); slots after the first must be a ride id that's a live Stage-B candidate at that stop (i.e. `legal: true` in that slot's distribution).
  - `force_slot=0`: pins the opener; the decoder then continues slots `1..K-1` under the tail mask (open ∧ unfinished ∧ not already picked). Slot-0 logits stay the natural policy — only the chosen action changes.
  - `force_slot>0`: slot 0 and any slots before the pin decode naturally first (this determines which rides are even reachable as Stage-B candidates); the pin only overrides the one requested stop. If the requested ride isn't a candidate there (e.g. it wasn't among the model's top-`candidate_m` picks conditioned on the natural opener), the API returns a 400 rather than silently ignoring the pin.
- Response includes `distributions_by_slot[k]` (softmax over that step's mask), `natural_recommended` (slot-0 argmax without any pin), and `forced_slot`/`forced_action`.
- The plan is a **deterministic** function of the request (prefs/must-dos/history/location/leave/arrival/model, all unchanged) — `recommend()` always decodes with `close_margin=0` (no close-call sampling) for both backends, specifically so a stop shown as pinnable in one response stays reachable if the guest immediately pins it in the next request.
- Models exported before this feature only support `force_slot=0` (`supports_force_any_slot: false` in `/api/catalog` and the recommend response's `model` block); the UI hides pinning for later stops on those models.
- UI: tap a legal stop anywhere in the route timeline to see its alternatives, then tap one to pin/unpin it. Force state is ephemeral (not in `localStorage`) and clears when switching model versions.

## Persistence

All guest state (prefs, must-dos, completions, location, leave time, undo/redo stacks) lives in the browser `localStorage` key `omniqueue-companion-v1`. Theme preference is stored separately under `omniqueue-companion-theme` and **defaults to dark** when unset. The API is stateless per request. Force-pick exploration is UI-only and is cleared when switching model versions.

## Notes

- Credit [ThemeParks.wiki](https://themeparks.wiki) for wait data; respect their rate guidance (companion caches ~45s).
- `run_companion.py` inserts the parent of `Park/` onto `sys.path` so you do not need to set `PYTHONPATH` (the editable wheel only installs `_park_sim`, not the Python package tree).
- Deploy image uses **onnxruntime** only (no PyTorch) so it fits Render’s free ~512 MB RAM tier.
- `Park.training.features` keeps dimension constants importable without torch; torch masking helpers load torch lazily for training.
- Walk times load from committed `data/walk_matrix.npz` (do not rely on gitignored `cache/` in deploy). Rebuild with `python -c "from Park.park_graph import get_park_graph, reset_park_graph; reset_park_graph(); get_park_graph(force_recompute=True)"` then copy `cache/walk_matrix.npz` → `data/walk_matrix.npz` if pathways/config change.
