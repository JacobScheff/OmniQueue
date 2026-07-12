# Rides and Queues

**Module:** `native/src/park_sim.cpp` (`Ride`, ride state arrays)

## Overview

34 rides with state managed in C++. Normal operation uses **implicit FIFO boarding** via scheduled `RideStart` / `RideComplete` events — no per-party physical queue simulation.

## Key Fields

| Field | Purpose |
|-------|---------|
| `capacity_per_sec` | Throughput derived from hourly capacity ÷ 3600 (from `graph_data.hpp`) |
| `next_board_sec` | Earliest second the next party can board |
| `pending_board` | Parties awaiting `RideStart` |
| `on_ride` | Parties currently riding |
| `current_wait` | Estimated wait for routing decisions |

## Boarding Flow

1. Party arrives at ride (OPEN) → compute `start_sec = max(now, next_board_sec)`.
2. Schedule `RideStart` at `start_sec`; advance `next_board_sec` by `1 / capacity_per_sec`.
3. On `RideStart` → party `ON_RIDE`; schedule `RideComplete` after `duration_sec`.
4. On `RideComplete` → update history, trigger routing.

## Wait Estimate

```
wait ≈ max(0, next_board_sec - now) + (pending + on_ride) / capacity_per_sec
```

Broken rides report wait ≥ 9000 for routing exclusion.
