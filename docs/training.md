# Training (Phase 2–3)

**Modules:** `model.py`, `training/`, `_park_sim.ParkEnv`

## Overview

1. **Phase 2 — Behavioral cloning:** mine heuristic routing decisions from the C++ simulator and train `ParkRouterModel` via cross-entropy loss.
2. **Phase 3 — PPO:** fine-tune the policy on **complete park days** (`ParkEnv` runs until the day ends, ~500k routing decisions/day). PPO trains on a random subsample of transitions for memory efficiency.

Checkpoints save automatically to the configured `save_dir` (`checkpoints/bc/` and `checkpoints/ppo/` by default).

## Configuration

All hyperparameters live in **`config.py`** under the `BC_*` and `PPO_*` sections. Edit that file to tune learning rates, discount factors, clip coefficients, batch sizes, and so on — no command-line flags needed.

The training scripts only accept the small number of **run-time parameters** that typically vary between experiments (seed, number of days, device, etc.).

## Prerequisites

```bash
pip install -e .
```

Requires the C++ extension (`_park_sim`) and PyTorch.

## Phase 2: Behavioral Cloning

```bash
python training/bc_train.py --seed 42 --bc-days 1 --device cpu
```

- Labels come from `_park_sim.collect_bc_dataset()` (~500k samples/day).
- Hyperparameters (`BC_EPOCHS`, `BC_LR`, `BC_BATCH_SIZE`, etc.) are set in `config.py`.
- Output: `checkpoints/bc/bc_final.pt`

## Phase 3: PPO

Warm-start from BC (recommended):

```bash
python training/ppo_train.py \
  --seed 42 \
  --init-checkpoint checkpoints/bc/bc_final.pt \
  --total-days 20 \
  --device cpu
```

`--init-checkpoint` loads **model weights only** into the PPO agent (fresh Adam). Replacing `agent.model` with a newly constructed module would orphan the optimizer and silently freeze training.
All PPO hyperparameters (`PPO_LEARNING_RATE`, `PPO_GAMMA`, `PPO_GAE_LAMBDA`, `PPO_CLIP_COEF`, etc.) are configured in `config.py`.

Each **update** simulates `--num-envs` full park days (8 AM–11 PM), then runs PPO on up to `PPO_SUBSAMPLE_SIZE` random routing decisions per day. Rollouts use the native C++ simulator with batched policy inference (`ParkEnv.exchange_batch`, batch size `PPO_INFERENCE_BATCH_SIZE`) so the DES stays in C++ and PyTorch runs once per batch instead of once per routing step.

Expect ~10–60 seconds per rollout day depending on hardware.

Output: `checkpoints/ppo/ppo_final.pt`

## PPO reward (ParkEnv)

Objective: **get each party’s preferred and must-do rides done quickly** (optionally guest-weighted by `party_size`). Wait variance is **not** in the training reward (still logged as a diagnostic KPI).

Rewards are emitted **only on routing decisions** (~300k–500k/day). Components (C++ `env_reward_delta` / terminal bonus):

| Component | When | Formula (defaults in `config.py`) |
|-----------|------|-----------------------------------|
| **Dense urgency** | **Every** routing step (party-local) | `-PPO_MUST_DO_URGENCY_COEF × remaining_must_dos` `- PPO_PREF_URGENCY_COEF × remaining_pref_mass` (`remaining_pref_mass` = Σ preference for rides with `history == 0`) |
| Preference / must-do | Party’s **next** routing step after a real `RideComplete` | `time_factor × (PPO_PREF_REWARD_SCALE × preference[ride] + PPO_MUST_DO_COMPLETION_BONUS if must-do)`; `time_factor = max(0, 1 - PPO_TIME_DECAY × (now − spawn) / DAY)`; × `party_size` if `PPO_WEIGHT_BY_PARTY_SIZE` |
| Terminal must-do | Last routing step | `-PPO_UNFULFILLED_MUST_DO_PENALTY × (remaining / assigned)` + flush leftover pending bonuses |

Preference / must-do bonuses are **accumulated at ride completion** into a per-party pending buffer and **flushed** when that party is routed again. Breakdown evacuations never call `RideComplete`, so they earn **no** preference credit (see `docs/breakdowns.md`).

Must-do completion bonuses dominate filler preference mass. Tune via `config.py` (mirrored in `native/include/park_sim.hpp`). See also `docs/pref-mustdo-reward-plan.md`.

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
| Guest features | `(B, G, 46)` | Prefs `0..33`, remaining pref mass `34`, party state `35..44`, elapsed since spawn `45` |
| Ride features | `(B, G, 34, 8)` | Wait, incoming, open, duration, capacity, walk, history, must-do |
| Env features | `(B, 4)` | Time of day, mean wait, **wait-variance slot zeroed**, broken fraction |
| Actions | `0–33` ride, `34` exit, `35` idle wander |

Flat observation size: **322** (`FLAT_OBS_DIM` = 46 + 34×8 + 4).

### Architecture (`ParkRouterModel`)

- **`d_model=256`**, **3** guest transformer layers (self-attention + FFN), **8** heads — the coordinator attends across parties in the same routing wave.
- Symmetric ride encoder: ride-id embedding + MLP over the 8 dynamic feats (no raw env concat on rides).
- Pointer scores = guest queries × per-party ride keys; exit/idle from a linear head.
- Per-party critic head.
- **Action masking** before CE / `Categorical`: closed rides, already-at ride, time-infeasible rides, and soft-close (exit-only) are illegal. BC uses **masked cross-entropy** over padded multi-party waves.

BC groups heuristic labels by `wave_id` (parties routed in the same `decide_routes` call), then **chunks** each wave to at most `MAX_COORDINATOR_GUESTS` (default **32**) so the coordinator never attends over opening-rush cohorts of thousands of parties. PPO uses the same chunk size. This cap is what prevents OOM — shrinking `d_model` alone does not, because BC collate previously padded every minibatch to the largest wave in the batch (G up to ~2800) and attention is O(G²).

Masked BC loss ignores padded guests entirely (never `0 * inf`) and always keeps the expert label unmasked so a slightly strict feasibility mask cannot send CE to `+inf`.

Related memory knobs in `config.py`:

| Knob | Default | Notes |
|------|---------|--------|
| `MAX_COORDINATOR_GUESTS` | 32 | Hard cap on G per forward |
| `BC_BATCH_SIZE` | 64 | Count of *waves*, not decisions |
| `PPO_INFERENCE_BATCH_SIZE` | 256 | C++ pending parties; policy chunks to `MAX_COORDINATOR_GUESTS` |
| `PPO_UPDATE_WAVE_BATCH` | 32 | Waves per optimizer step (small = less laptop display freeze) |
| `PPO_UPDATE_YIELD_SEC` | 0.05 | Sleep after each update step so the desktop can refresh |

## Checkpoint format

```python
from training.checkpoint import load_checkpoint
model, step, extra = load_checkpoint("checkpoints/bc/bc_final.pt")
```

Each `.pt` file has a sibling `.json` with step metadata.
