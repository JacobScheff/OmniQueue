# Parties

**Module:** `native/src/park_sim.cpp` (`PartyArrays`, `spawn_day`)

## Overview

Guests are grouped into **parties** stored in a **struct-of-arrays** layout inside the C++ simulator.

## Core Arrays

| Array | dtype | Description |
|-------|-------|-------------|
| `leave_sec` | int32 | Scheduled park departure second |
| `location_node_idx` | int32 | Current graph node index |
| `effective_speed` | float32 | Party walking speed (min-of-member draw) |
| `preference_order` | int16 × N | Pre-sorted ride indices per party (`N = NUM_RIDES`) |
| `preferences` | float32 × N | Normalized per-ride preference masses |
| `balk_sec` | float32 × N | Precomputed balk thresholds |
| `must_do_remaining` | uint8 × N | Unfinished must-do flags |
| `ride_history` | int16 × N | Completions per ride (drives heuristic repeat dampening) |
| `rides_completed` | int32 | Total completions for this party |
| `walk_target_ride` | int32 | Ride id while walking (-1 if none) |
| `state` | int8 | `PartyState` bitmask value |

Spawn constants live in `native/include/park_sim.hpp` (mirrored from `config.py`).

## Spawn Model

| Parameter | Default |
|-----------|---------|
| Total guests/day | ~50,000 ± 2,500 |
| Party size | `max(1, round(N(3.2, 1.0)))`, no cap |
| Arrival mixture | **65%** rope-drop rush (`N(8 min, 12 min)`, clamped to first 2 h); **35%** daytime (`N(6 h, 3.5 h)`) |
| Dwell time | Mean **14 h**, σ = 2.5 h, min 2 h |
| Leave time | `min(DAY_SECONDS, spawn + dwell)` — many early arrivals stay until official close |

### Soft park close

Official close is `DAY_SECONDS` (11:00 PM). After close:

- Routing assigns **exit only** (no new rides or idle wander).
- Parties already **in queue** keep boarding; parties **on ride** finish.
- Walkers who arrive at a ride after close do **not** board — they re-route to exit.
- The timing wheel continues for up to `CLOSE_DRAIN_SEC` (3 h) so queued/on-ride parties can finish and walk out.

Parties whose `leave_sec` is the day end may still join long lines near close: feasibility uses the post-close drain window so a ride that finishes after official close remains allowed.

## Must-Do Lists

- Count per party: uniform **0–4** rides.
- Unfinished must-dos sort first in `preference_order`.
- Completing a must-do ride clears that flag, refreshes `preference_order` / `balk_sec`, and (in `ParkEnv` / PPO only) adds `PPO_MUST_DO_COMPLETION_BONUS` to the party’s pending preference reward (see `docs/training.md`).

## Balk Thresholds

```
balk_sec[r] = BASE_BALK_SEC + BALK_SCALE × preference[r] ** BALK_PREF_EXP
```

Defaults (`config.py` / `park_sim.hpp`): **40 min** base + up to **5 min** from preference (`BALK_SCALE`), so thresholds sit around **40–45 minutes**. Precomputed at spawn and refreshed when a must-do ride is completed.
