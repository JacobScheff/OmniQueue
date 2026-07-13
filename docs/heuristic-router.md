# Heuristic Router

**Module:** `native/src/park_sim.cpp` (`route_one`, `route_batch`)

## Overview

Phase 1 baseline router using **preference-ordered balking with ride-repeat dampening**, implemented in C++. Selected when `config.ROUTER = "heuristic"` (the default for `simulator.run_day()`).

Uses per-party `ride_history[ride]` (completions so far) so parties prefer untried rides and only repeat when preference or wait conditions justify it.

## Selection Algorithm

Shared feasibility checks for every candidate: not the ride the party is already at, ride open, and walk + wait + duration fits remaining park time.

Then choose the first matching pass:

### Pass 1 — Fresh rides

Iterate `preference_order`. Accept the first ride with `ride_history == 0` and `wait ≤ balk_sec[ride]`.

Unfinished must-dos sort first in `preference_order` and almost always have zero history, so they keep priority.

### Pass 2 — Preferred / limited repeats

Iterate `preference_order`. Accept a ride with `history ≥ 1` only if:

1. Preference gate: rank is among the top `REPEAT_TOP_K` (default 3), **or** normalized `preference[ride] ≥ REPEAT_PREF_THRESHOLD` (default 0.04).
2. Repeat budget: `history < max_repeats`, where

   ```
   max_repeats = min(REPEAT_MAX, 1 + floor(REPEAT_PREF_SCALE × preference × NUM_RIDES))
   ```

3. `wait ≤ balk_sec[ride] × REPEAT_BALK_FACTOR` (default factor 1.0).

Former must-dos keep elevated preference after completion (`MUST_DO_PREF_BOOST`), so a limited re-ride remains plausible.

### Pass 3 — Opportunistic short wait

Among feasible rides, accept short waits even with history:

- `wait ≤ SHORT_WAIT_SEC` (default 12 min), or
- the best feasible wait is itself ≤ `SHORT_WAIT_SEC` and this wait is within `SHORT_WAIT_SLACK_SEC` of that best.

If several qualify, pick lowest `ride_history`, then earlier preference rank.

### Pass 4 — Fallback

If no ride passes:

| Probability | Action |
|-------------|--------|
| 50% | Idle wander (`ROUTE_IDLE_CODE`) → random node within 2 hops |
| 50% | Force-pick first feasible ride (ignore balk), preferring `history == 0`, then lowest history |

Knobs live in `config.py` and are mirrored in `native/include/park_sim.hpp`.

## Batch Execution

- Parties routed in chunks of `MAX_ROUTE_BATCH` (256).
- Walk times read from `graph_data::kBaseWalkToRides[node_idx, ride_id]`.

## Usage

```python
from simulator import run_day

metrics = run_day(seed=0)  # heuristic routing in C++
```

PPO routing (`config.ROUTER = "ppo"`) is Phase 3 and not yet wired to the native simulator.

Behavioral cloning mines this heuristic; after router changes, re-run `training/bc_train.py` before treating old BC checkpoints as matching the expert.
