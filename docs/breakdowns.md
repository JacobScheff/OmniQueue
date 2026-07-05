# Breakdowns

**Module:** `rides.py`, `simulator.py`

## Overview

Rides stochastically break down each simulated second with probability `breakdown_prob_sec` (derived from hourly rate). While broken, boarding stops and parties evacuate in order.

## On Breakdown

| Party location | Behavior |
|----------------|----------|
| Walking to ride | **Immediate re-route** from current node |
| In queue (`pending_board`) | **Immediate routing decision** at ride entrance; added to evacuation deque |
| On ride | Added to `evacuating_on_ride`; evacuates **after** queue is empty |

## Evacuation

- Rate: **1 party every 4 seconds** (`EVAC_INTERVAL_SEC`).
- Queue parties (`evacuation` deque) leave first.
- On-ride parties (`evacuating_on_ride`) leave last.
- **`RIDE_COMPLETE` is not fired** for aborted on-ride parties — no ride credit.
- Each evacuated party routes at the ride entrance node when their `EVACUATE_PARTY` event fires.

## Reopening

- `broken_until_sec = now + Uniform(15 min, 60 min)`.
- `BREAKDOWN_END` event restores `OPEN` status and resets `next_board_sec`.

## Stale Events

`ride.generation` increments on breakdown. `RIDE_START` events with outdated generation are ignored.
