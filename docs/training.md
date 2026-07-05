# Training (Phase 2–3)

**Modules:** `model.py`, `training/`, `_park_sim.ParkEnv`

## Overview

1. **Phase 2 — Behavioral cloning:** mine heuristic routing decisions from the C++ simulator and train `ParkRouterModel` via cross-entropy loss.
2. **Phase 3 — PPO:** fine-tune the policy in `ParkEnv` (one routing decision per step) with a CleanRL-style PPO loop.

Checkpoints save automatically to `--save-dir` during training (`bc_step_*.pt`, `bc_final.pt`, `ppo_step_*.pt`, `ppo_final.pt`).

## Prerequisites

```bash
pip install -e .
```

Requires the C++ extension (`_park_sim`) and PyTorch.

## Phase 2: Behavioral Cloning

```bash
python training/bc_train.py \
  --seed 42 \
  --bc-days 1 \
  --epochs 3 \
  --batch-size 512 \
  --save-dir checkpoints/bc \
  --save-every 500
```

- Labels come from `_park_sim.collect_bc_dataset()` (~500k samples/day).
- Output: `checkpoints/bc/bc_final.pt`

## Phase 3: PPO

Warm-start from BC (recommended):

```bash
python training/ppo_train.py \
  --seed 42 \
  --init-checkpoint checkpoints/bc/bc_final.pt \
  --total-timesteps 500000 \
  --num-envs 4 \
  --save-dir checkpoints/ppo \
  --save-every 10000
```

Output: `checkpoints/ppo/ppo_final.pt`

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
| Guest features | `(B, 1, 45)` | Preferences + party state |
| Ride features | `(B, 35, 5)` | Wait, incoming, open, duration, capacity |
| Env features | `(B, 4)` | Time of day, mean wait, variance, broken fraction |
| Actions | `0–34` ride, `35` exit, `36` idle wander |

Flat observation size: **224** (`FLAT_OBS_DIM`).

## Checkpoint format

```python
from training.checkpoint import load_checkpoint
model, step, extra = load_checkpoint("checkpoints/bc/bc_final.pt")
```

Each `.pt` file has a sibling `.json` with step metadata.
