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

Rewards are emitted **only on routing decisions** (~300k–500k/day). Components (C++ `env_reward_delta` / terminal bonus):

| Component | When | Formula (defaults in `config.py`) |
|-----------|------|-----------------------------------|
| **Dense wait variance** | **Every** routing step | `-PPO_WAIT_VAR_STEP_COEF × current_wait_variance / 1e6` (fallback `-0.001` if no valid waits) |
| Preference / must-do | Party’s **next** routing step after a real `RideComplete` | `PPO_PREF_REWARD_SCALE × preference[ride]` (+ `PPO_MUST_DO_COMPLETION_BONUS` if that ride was a must-do) |
| Terminal wait variance | Last routing step of the day | `-avg_wait_variance / 1000` |
| Terminal must-do | Last routing step | `-PPO_UNFULFILLED_MUST_DO_PENALTY × remaining_must_dos` (park-wide) |

`current_wait_variance` is computed from live ride wait estimates at the routing step (same definition as KPI samples). The 300 s metrics sampler still records wait variance for logging / terminal bonus, but **no longer gates** the per-step reward — so GAE can credit load-balancing decisions within a few hundred routing steps instead of waiting ~5 park minutes.

Preference bonuses are **accumulated at ride completion** into a per-party pending buffer and **flushed** when that party is routed again. Breakdown evacuations never call `RideComplete`, so they earn **no** preference credit (see `docs/breakdowns.md`).

Preference scales are intentionally **secondary** to wait variance (`PPO_PREF_REWARD_SCALE=0.01`, must-do `0.005`). Tune via `config.py` (mirrored in `native/include/park_sim.hpp`).

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
| Guest features | `(B, G, 45)` | Preferences + party state (`G` = co-timed parties in a routing wave) |
| Ride features | `(B, G, 34, 8)` | Wait, incoming, open, duration, capacity, walk, history, must-do |
| Env features | `(B, 4)` | Time of day, mean wait, variance, broken fraction |
| Actions | `0–33` ride, `34` exit, `35` idle wander |

Flat observation size: **321** (`FLAT_OBS_DIM` = 45 + 34×8 + 4).

### Architecture (`ParkRouterModel`)

- **`d_model=256`**, **3** guest transformer layers (self-attention + FFN), **8** heads — the coordinator attends across parties in the same routing wave.
- Symmetric ride encoder: ride-id embedding + MLP over the 8 dynamic feats (no raw env concat on rides).
- Pointer scores = guest queries × per-party ride keys; exit/idle from a linear head.
- Per-party critic head.
- **Action masking** before CE / `Categorical`: closed rides, already-at ride, time-infeasible rides, and soft-close (exit-only) are illegal. BC uses **masked cross-entropy** over padded multi-party waves.

BC groups heuristic labels by `wave_id` (parties routed in the same `decide_routes` call), then **chunks** each wave to at most `MAX_COORDINATOR_GUESTS` (default **32**) so the coordinator never attends over opening-rush cohorts of thousands of parties. PPO uses the same chunk size. This cap is what prevents OOM — shrinking `d_model` alone does not, because BC collate previously padded every minibatch to the largest wave in the batch (G up to ~2800) and attention is O(G²).

Related memory knobs in `config.py`:

| Knob | Default | Notes |
|------|---------|--------|
| `MAX_COORDINATOR_GUESTS` | 32 | Hard cap on G per forward |
| `BC_BATCH_SIZE` | 64 | Count of *waves*, not decisions |
| `PPO_INFERENCE_BATCH_SIZE` | 256 | C++ pending parties; policy chunks to `MAX_COORDINATOR_GUESTS` |

## Checkpoint format

```python
from training.checkpoint import load_checkpoint
model, step, extra = load_checkpoint("checkpoints/bc/bc_final.pt")
```

Each `.pt` file has a sibling `.json` with step metadata.
