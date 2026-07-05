# Heuristic Router

**Module:** `router/heuristic.py`, `router/numba_routing.py`

## Overview

Phase 1 baseline router using **preference-ordered balking**, accelerated with a **Numba `@njit` kernel** (`route_batch_numba`). Selectable via `config.ROUTER = "heuristic"`.

## Selection Algorithm

For each party, iterate `preference_order` and pick the **first** ride where:

1. Not already at that ride, ride is open, enough time remains.
2. `current_wait_sec ≤ balk_sec[ride]`.

If no ride passes:

| Probability | Action |
|-------------|--------|
| 50% | Idle wander (`ROUTE_IDLE_CODE`) → random node within 2 hops |
| 50% | Force-pick first feasible ride (ignore balk) |

## Batch Execution

- Parties routed in chunks of `MAX_ROUTE_BATCH` (256).
- Walk times read from `graph.base_walk_to_rides[node_idx, ride_id]` inside Numba (no Python per-ride calls).

## Switching Routers

```python
config.ROUTER = "heuristic"  # or "ppo" (Phase 3)
run_day(seed=0, router="heuristic")
```

## Dependencies

Requires `numba` for full speed (see `requirements.txt`). If Numba is not installed, the same kernel runs as pure Python automatically (correct but slower).
