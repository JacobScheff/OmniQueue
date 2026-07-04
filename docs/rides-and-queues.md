# Rides and Queues

**Module:** `rides.py`

## Overview

35 rides with vectorized state managed by `RideManager`. Normal operation uses **implicit FIFO boarding** via scheduled `RIDE_START` / `RIDE_COMPLETE` events — no per-party physical queue simulation.

## Key Fields

| Field | Purpose |
|-------|---------|
| `capacity_per_sec` | Throughput derived from hourly capacity ÷ 3600 |
| `next_board_sec` | Earliest second the next party can board |
| `pending_board` | `{party_id: scheduled_start_sec}` awaiting `RIDE_START` |
| `on_ride` | Parties currently riding |
| `current_wait_sec` | Estimated wait for routing decisions |

## Boarding Flow

1. Party arrives at ride (OPEN) → compute `start_sec = max(now, next_board_sec)`.
2. Schedule `RIDE_START` at `start_sec`; advance `next_board_sec` by `1 / capacity_per_sec`.
3. On `RIDE_START` → party `ON_RIDE`; schedule `RIDE_COMPLETE` after `duration_sec`.
4. On `RIDE_COMPLETE` → update history, trigger routing.

## Wait Estimate

```
wait ≈ max(0, next_board_sec - now) + (pending + on_ride) / capacity_per_sec
```

Broken rides report wait = 9999.
