# Training (Phase 2–3)

**Modules:** `model.py`, `training/`, `_park_sim.ParkEnv`

## Overview

1. **Phase 2 — Behavioral cloning:** mine heuristic routing decisions from the C++ simulator and train `ParkRouterModel` via cross-entropy on **slot-0** logits (flat single-party batches).
2. **Phase 3 — PPO (personal planner):** fine-tune on complete park days where **`PPO_NUM_FOCALS` parties** are PPO-controlled and the rest of the park is heuristic-routed. The policy emits a length-**`PPO_ROUTE_K`** ride route; only `route[0]` is applied in the DES. Rewards / KPIs are focal-only.

Checkpoints save automatically to the configured `save_dir` (`checkpoints/bc/` and `checkpoints/ppo/` by default).

## Configuration

All hyperparameters live in **`config.py`** under the `BC_*` and `PPO_*` sections. Edit that file to tune learning rates, discount factors, clip coefficients, batch sizes, and so on — no command-line flags needed.

The training scripts only accept the small number of **run-time parameters** that typically vary between experiments (seed, number of days, device, focals, etc.).

## Prerequisites

From the `Park/` directory:

```bash
pip install -e .
```

Requires the C++ extension (`_park_sim`) and PyTorch.

Training scripts bootstrap `sys.path` so `import Park.*` works when launched as
`python training/ppo_train.py` (or `bc_train.py`) with cwd = `Park/`.

## Phase 2: Behavioral Cloning

```bash
python training/bc_train.py --seed 42 --bc-days 1 --device cpu
```

- Labels come from `_park_sim.collect_bc_dataset()` (~500k samples/day).
- Each sample is one party decision; batches are flat `(B, …)` (no guest axis).
- Hyperparameters (`BC_EPOCHS`, `BC_LR`, `BC_BATCH_SIZE`, etc.) are set in `config.py`.
- Output: `checkpoints/bc/bc_final.pt`

## Phase 3: PPO (personal focals)

Warm-start from BC (optional; encoder tensors load flexibly into the route decoder model):

```bash
python training/ppo_train.py \
  --seed 42 \
  --init-checkpoint checkpoints/bc/bc_final.pt \
  --total-days 20 \
  --num-focals 24 \
  --device cpu
```

`--init-checkpoint` loads matching weight tensors into the PPO agent (fresh Adam). `rank_route_v1` has **no legacy widen bridges** — use a fresh BC checkpoint trained on the current obs layout. Checkpoints carry `arch_version=rank_route_v1`. See [`rank-route-architecture.md`](rank-route-architecture.md).

Each day calls `ParkEnv.reset_personal(seed, n_focals)`:

- Spawns a full park (~50k guests) with **training-only randomized** prefs/must-dos (play/watch/visualize keep popularity-weighted spawn — see `docs/parties.md`).
- Marks N parties as size-1 focals (spread across spawn times).
- Heuristic-routes the crowd; only focals enter the PPO `env_queue_`.
- `exchange_batch` returns focal observations (+ `party_ids`); GAE breaks when `party_id` changes.

BC mining (`collect_bc_dataset`) also uses the randomized training spawn so expert labels match the personal-planner preference distribution.

**Checkpoint compatibility:** the personal/no-G architecture does not load old coordinator checkpoints. Retrain from scratch (or from a matching BC checkpoint).

Output: `checkpoints/ppo/ppo_final.pt`

## PPO reward (focal learners)

Objective: **get each focal party’s preferred and must-do rides done quickly**. Wait variance is **not** in the training reward.

Rewards are emitted **only on focal routing decisions**. Components:

| Component | When | Formula (defaults in `config.py`) |
|-----------|------|-----------------------------------|
| **Dense urgency** | Every focal routing step | `-PPO_MUST_DO_URGENCY_COEF × remaining_must_dos` `- PPO_PREF_URGENCY_COEF × Σ preference[r]**PPO_PREF_REWARD_EXP` (unfinished only) |
| Preference / must-do | Focal’s next routing step after a real `RideComplete` | `time_factor × (PPO_PREF_REWARD_SCALE × preference[ride]**PPO_PREF_REWARD_EXP + PPO_MUST_DO_COMPLETION_BONUS if must-do)` |
| Terminal must-do | Last routing step of the day | `-PPO_UNFULFILLED_MUST_DO_PENALTY × (focal_remaining / focal_assigned)` + flush leftover **focal** pending bonuses |
| Planned walk | Python shaping at emission | `-PPO_ROUTE_PLANNED_WALK_COEF × mean_inter_ride_walk / PPO_ROUTE_WALK_NORM_SEC` |
| Realized walk | Python shaping with the reward | `-PPO_ROUTE_REALIZED_WALK_COEF × walk_to_commit / PPO_ROUTE_WALK_NORM_SEC` |

Crowd completions do not write into the pending preference buffer when hybrid personal mode is active. Route-consistency shaping and party-size reward weighting are **removed** in `rank_route_v1`.

**Counterfactual regularizers (training only, Stage A):** hinge Jensen–Shannon under (1) resampled prefs/must-dos (`PPO_CF_*`) and (2) perturbed waits (`PPO_CF_WAIT_*`). See [`rank-route-architecture.md`](rank-route-architecture.md).

**Stage A preference ranking (training only):** soft CE of Stage A toward unfinished sharpened pref (+ must-do bonus) via `PPO_PREF_RANK_COEF`. Primary commit reward still only uses `route[0]`.

**Path-conditioned walk:** during Stage B decode, after each ride pick the model rewrites walk/ETA as if the party were at that ride and re-encodes candidate keys. Slot 0 (Stage A) still uses observation walks from the true current location.

## Evaluate a checkpoint

```bash
python training/eval_policy.py \
  --checkpoint checkpoints/ppo/ppo_final.pt \
  --episodes 3 \
  --seed 42
```

## Model I/O

| Tensor | Shape | Content |
|--------|-------|---------|
| Guest features | `(B, 43)` | Prefs `0..33`, remaining pref mass `34`, speed `35`, time left `36`, loc `37`, rides completed `38`, must-do count `39`, at-ride `40`, state `41`, elapsed `42` |
| Ride features | `(B, 34, 11)` | Wait, incoming, open, duration, capacity, walk, history, must-do, unfinished pref, ETA, wait_vs_mean |
| Env features | `(B, 3)` | Time of day, mean wait, broken fraction |
| Route actions | `(B, K)` | `K=PPO_ROUTE_K` (default 5); rides `0–33`, or exit `34` / idle `35` in slot 0 only; later slots `-1` pad after exit/idle |
| Commit | scalar | `route[0]` applied in the DES / companion |

Flat observation size: **420** (`FLAT_OBS_DIM` = 43 + 34×11 + 3).

### Architecture (`RankRouteModel` / `ParkRouterModel` alias)

See [`rank-route-architecture.md`](rank-route-architecture.md): `d_model=384`, 4-head guest→ride cross-attn, Stage A MLP scorer, top-M candidate Stage B GRU+pointer. Inference may sample Stage A when top-2 probs are within `INFER_CLOSE_MARGIN`.

Related knobs in `config.py`:

| Knob | Default | Notes |
|------|---------|--------|
| `PPO_NUM_FOCALS` | 24 | Focal parties per training day |
| `PPO_ROUTE_K` | 5 | Emitted route length |
| `PPO_CANDIDATE_M` | 8 | Stage B candidate set size |
| `PPO_ENT_COEF` | 0.03 | Entropy bonus (Stage A + slot-weighted decoder) |
| `PPO_CF_COEF` / `MARGIN` / `FRAC` | `0.1` / `0.15` / `0.25` | Pref counterfactual JS hinge |
| `PPO_CF_WAIT_*` | see config | Wait counterfactual JS hinge |
| `PPO_PREF_RANK_COEF` | `0.025` | Soft pref ranking on Stage A |
| `INFER_TEMP` / `TOP_P` / `CLOSE_MARGIN` | `0.8` / `0.9` / `0.12` | Close-call sampling |
| `BC_BATCH_SIZE` | 256 | Individual decisions per BC minibatch |
| `PPO_INFERENCE_BATCH_SIZE` | 256 | Max pending focals per `exchange_batch` |
| `PPO_UPDATE_MB_SIZE` | 256 | Transitions per optimizer step |
| `PPO_UPDATE_YIELD_SEC` | 0.05 | Sleep after each update step |

## Checkpoint format

```python
from training.checkpoint import load_checkpoint
model, step, extra = load_checkpoint("checkpoints/bc/bc_final.pt")
```

Each `.pt` file has a sibling `.json` with step metadata.
