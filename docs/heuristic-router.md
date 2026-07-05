# Heuristic Router

**Module:** `router/heuristic.py`, `router/base.py`

## Overview

Phase 1 baseline router using **preference-ordered balking**. Selectable via `config.ROUTER = "heuristic"`. The PPO router (`router/ppo.py`) raises `NotImplementedError` until Phase 3.

## Selection Algorithm

For each party, iterate rides in `preference_order` (must-dos first, then descending preference):

1. Skip if party is already at that ride, ride is closed, or insufficient time remains.
2. Pick the **first** ride where `current_wait_sec ≤ balk_sec[ride]`.

If no ride passes:

| Probability | Action |
|-------------|--------|
| 50% | Idle wander to random node within 2 hops |
| 50% | Force-pick first feasible ride (ignore balk threshold) |

If force-pick fails → exit park (`EXIT_RIDE_ID = -1`).

## Balk Formula

```python
balk_sec = BASE_BALK_SEC + BALK_SCALE × preference ** BALK_PREF_EXP
```

Defaults: `BASE=600`, `SCALE=2400`, `EXP=1.5`.

## Switching Routers

```python
# config.py
ROUTER = "heuristic"  # or "ppo" (Phase 3)

# programmatic
from simulator import run_day
run_day(seed=0, router="heuristic")
```

Both routers must implement `Router.route_batch()` in `router/base.py`.
