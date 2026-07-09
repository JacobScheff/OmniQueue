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
| `preference_order` | int16 × 35 | Pre-sorted ride indices per party |
| `balk_sec` | float32 × 35 | Precomputed balk thresholds |
| `walk_target_ride` | int32 | Ride id while walking (-1 if none) |
| `state` | int8 | `PartyState` bitmask value |

Spawn constants live in `native/include/park_sim.hpp` (mirrored from `config.py`).

## Spawn Model

| Parameter | Default |
|-----------|---------|
| Total guests/day | ~50,000 ± 2,500 |
| Party size | `max(1, round(N(3.2, 1.0)))`, no cap |
| Arrival peak | ~11:00 AM (bell curve) |
| Dwell time | Mean **10 h**, σ = 2 h, min 2 h |

## Must-Do Lists

- Count per party: uniform **0–4** rides.
- Unfinished must-dos sort first in `preference_order`.
- Completing a must-do ride clears that flag, refreshes `preference_order` / `balk_sec`, and (in `ParkEnv` / PPO only) adds `PPO_MUST_DO_COMPLETION_BONUS` to the party’s pending preference reward (see `docs/training.md`).

## Balk Thresholds

```
balk_sec[r] = BASE_BALK_SEC + BALK_SCALE × preference[r] ** BALK_PREF_EXP
```

Defaults (`config.py` / `park_sim.hpp`): **40 min** base + up to **5 min** from preference (`BALK_SCALE`), so thresholds sit around **40–45 minutes**. Precomputed at spawn and refreshed when a must-do ride is completed.
