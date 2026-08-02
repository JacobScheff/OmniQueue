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

`--init-checkpoint` loads matching weight tensors into the PPO agent (fresh Adam). Old single-action checkpoints warm-start the shared encoder / pointer projections; the GRU decoder starts fresh. An 8-feat ride encoder widens into the current 9-feat first linear (new unfinished-pref column zeroed). Checkpoints carry `arch_version=route_v1`.

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
| Route consistency | Python shaping on the transition | `PPO_ROUTE_CONSIST_COEF × Σ w_i · 1[new[i]==prev[i+1]]` (front-weighted; skips illegal) |
| Planned walk | Python shaping at emission | `-PPO_ROUTE_PLANNED_WALK_COEF × mean_inter_ride_walk / PPO_ROUTE_WALK_NORM_SEC` |
| Realized walk | Python shaping with the reward | `-PPO_ROUTE_REALIZED_WALK_COEF × walk_to_commit / PPO_ROUTE_WALK_NORM_SEC` |

Crowd completions do not write into the pending preference buffer when hybrid personal mode is active.

**Anti-collapse regularizer (training only):** hinge Jensen–Shannon between slot-0 distributions under the real pref/must-do vector vs a counterfactual resample (`PPO_CF_COEF`, `PPO_CF_MARGIN`, `PPO_CF_FRAC`). See `docs/route-plan-output-plan.md`.

**Tail preference ranking (training only):** soft cross-entropy on early route-tail slots (`PPO_PREF_RANK_SLOTS`, default 1–2) toward unfinished sharpened pref (+ must-do bonus), masked by each slot’s legal rides (`PPO_PREF_RANK_COEF`). Primary commit reward still only uses `route[0]`.

**Path-conditioned walk:** during AR decode, after each ride pick the model rewrites `RIDE_FEAT_WALK` as if the party were at that ride (inter-ride matrix buffer) and re-encodes ride keys for the next slot. Slot 0 still uses observation walks from the true current location.

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
| Guest features | `(B, 46)` | Prefs `0..33`, remaining sharpened pref mass `34` (`Σ pref**PPO_PREF_REWARD_EXP` unfinished), party state `35..44`, elapsed since spawn `45` |
| Ride features | `(B, 34, 9)` | Wait, incoming, open, duration, capacity, walk, history, must-do, unfinished sharpened pref (`pref**PPO_PREF_REWARD_EXP`, else 0 if already ridden) |
| Env features | `(B, 4)` | Time of day, mean wait, **wait-variance slot zeroed**, broken fraction |
| Route actions | `(B, K)` | `K=PPO_ROUTE_K` (default 6); rides `0–33`, or exit `34` / idle `35` in slot 0 only; later slots `-1` pad after exit/idle |
| Commit | scalar | `route[0]` applied in the DES / companion |

Flat observation size: **356** (`FLAT_OBS_DIM` = 46 + 34×9 + 4).

Warm-start from an 8-feat checkpoint: `--init-checkpoint` loads matching tensors and **widens** `ride_feat_proj.0` (copies old columns, zeros the new unfinished-pref column). Fresh Adam.

### Architecture (`ParkRouterModel`)

- **`d_model=256`** single-party encoder (no guest transformer / no G axis).
- Ride encoder: ride-id embedding + MLP over the 9 dynamic feats.
- **Autoregressive pointer decoder:** slot 0 uses guest query × ride keys + exit/idle head; slots `1..K-1` continue with a GRU cell + no-replacement ride pointer (open + unfinished only), with walk features refreshed along the planned path.
- Critic head over guest + mean ride embedding + env (single scalar; decision-time encode only).
- **Action masking** before CE / `Categorical`: closed rides, already-at ride, time-infeasible rides, and soft-close (exit-only) are illegal on slot 0.
- Entropy bonus uses early-slot weights (`PPO_ROUTE_ENTROPY_WEIGHTS`); no softmax temperature annealing.

Related knobs in `config.py`:

| Knob | Default | Notes |
|------|---------|--------|
| `PPO_NUM_FOCALS` | 24 | Focal parties per training day |
| `PPO_ROUTE_K` | 6 | Emitted route length |
| `PPO_ENT_COEF` | 0.03 | Entropy bonus (slot-weighted) |
| `PPO_CF_COEF` / `MARGIN` / `FRAC` | `0.1` / `0.15` / `0.25` | Counterfactual pref JS hinge |
| `PPO_PREF_RANK_COEF` / `SLOTS` | `0.05` / `(1, 2)` | Soft pref ranking on early tail slots |
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
