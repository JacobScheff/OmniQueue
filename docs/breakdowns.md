# Breakdowns

**Module:** `native/src/park_sim.cpp`

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
- On-ride parties evacuate last; they do **not** receive ride completion credit.
- After repair, `BreakdownEnd` reopens the ride.

## Repair

Uniform random duration between 15 and 60 minutes (`BREAKDOWN_REPAIR_MIN_SEC` / `MAX` in `park_sim.hpp`).
