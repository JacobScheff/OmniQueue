# Training (Phase 2–3)

**Modules:** `model.py`, `training/`, `_park_sim.ParkEnv`

## Overview

1. **Phase 2 — Behavioral cloning:** mine heuristic routing decisions from the C++ simulator and train `ParkRouterModel` via cross-entropy loss (flat single-party batches).
2. **Phase 3 — PPO (personal planner):** fine-tune on complete park days where **`PPO_NUM_FOCALS` parties** are PPO-controlled and the rest of the park is heuristic-routed. Rewards / KPIs are focal-only.

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

Warm-start from BC (optional; architecture must match — retrain BC after the no-G model change):

```bash
python training/ppo_train.py \
  --seed 42 \
  --init-checkpoint checkpoints/bc/bc_final.pt \
  --total-days 20 \
  --num-focals 24 \
  --device cpu
```

`--init-checkpoint` loads **model weights only** into the PPO agent (fresh Adam).

Each day calls `ParkEnv.reset_personal(seed, n_focals)`:

- Spawns a full park (~50k guests) with **randomized** prefs/must-dos.
- Marks N parties as size-1 focals (spread across spawn times).
- Heuristic-routes the crowd; only focals enter the PPO `env_queue_`.
- `exchange_batch` returns focal observations (+ `party_ids`); GAE breaks when `party_id` changes.

**Checkpoint compatibility:** the personal/no-G architecture does not load old coordinator checkpoints. Retrain from scratch (or from a matching BC checkpoint).

Output: `checkpoints/ppo/ppo_final.pt`

## PPO reward (focal learners)

Objective: **get each focal party’s preferred and must-do rides done quickly**. Wait variance is **not** in the training reward.

Rewards are emitted **only on focal routing decisions**. Components:

| Component | When | Formula (defaults in `config.py`) |
|-----------|------|-----------------------------------|
| **Dense urgency** | Every focal routing step | `-PPO_MUST_DO_URGENCY_COEF × remaining_must_dos` `- PPO_PREF_URGENCY_COEF × remaining_pref_mass` |
| Preference / must-do | Focal’s next routing step after a real `RideComplete` | `time_factor × (PPO_PREF_REWARD_SCALE × preference[ride] + PPO_MUST_DO_COMPLETION_BONUS if must-do)` |
| Terminal must-do | Last routing step of the day | `-PPO_UNFULFILLED_MUST_DO_PENALTY × (focal_remaining / focal_assigned)` + flush leftover **focal** pending bonuses |

Crowd completions do not write into the pending preference buffer when hybrid personal mode is active.

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
| Guest features | `(B, 46)` | Prefs `0..33`, remaining pref mass `34`, party state `35..44`, elapsed since spawn `45` |
| Ride features | `(B, 34, 8)` | Wait, incoming, open, duration, capacity, walk, history, must-do |
| Env features | `(B, 4)` | Time of day, mean wait, **wait-variance slot zeroed**, broken fraction |
| Actions | `0–33` ride, `34` exit, `35` idle wander |

Flat observation size: **322** (`FLAT_OBS_DIM` = 46 + 34×8 + 4).

### Architecture (`ParkRouterModel`)

- **`d_model=256`** single-party encoder (no guest transformer / no G axis).
- Ride encoder: ride-id embedding + MLP over the 8 dynamic feats.
- Pointer scores = guest query × ride keys; exit/idle from a linear head.
- Critic head over guest + mean ride embedding + env.
- **Action masking** before CE / `Categorical`: closed rides, already-at ride, time-infeasible rides, and soft-close (exit-only) are illegal.

Related knobs in `config.py`:

| Knob | Default | Notes |
|------|---------|--------|
| `PPO_NUM_FOCALS` | 24 | Focal parties per training day |
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
