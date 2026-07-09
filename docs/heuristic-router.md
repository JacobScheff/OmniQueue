# Heuristic Router

**Module:** `native/src/park_sim.cpp` (`route_one`, `route_batch`)

## Overview

Phase 1 baseline router using **preference-ordered balking**, implemented in C++. Selected when `config.ROUTER = "heuristic"` (the default for `simulator.run_day()`).

## Selection Algorithm

For each party, iterate `preference_order` and pick the **first** ride where:

1. Not already at that ride, ride is open, enough time remains.
2. `current_wait_sec ≤ balk_sec[ride]` (defaults ~40–45 min; see `docs/parties.md`).

If no ride passes:

| Probability | Action |
|-------------|--------|
| 50% | Idle wander (`ROUTE_IDLE_CODE`) → random node within 2 hops |
| 50% | Force-pick first feasible ride (ignore balk) |

## Batch Execution

- Parties routed in chunks of `MAX_ROUTE_BATCH` (256).
- Walk times read from `graph_data::kBaseWalkToRides[node_idx, ride_id]`.

## Usage

```python
from simulator import run_day

metrics = run_day(seed=0)  # heuristic routing in C++
```

PPO routing (`config.ROUTER = "ppo"`) is Phase 3 and not yet wired to the native simulator.
