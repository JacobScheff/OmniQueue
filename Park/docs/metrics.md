# Metrics

**Modules:** `metrics.py` (`DayMetrics`), `native/src/park_sim.cpp` (collection)

## Overview

KPIs are sampled every `METRICS_SAMPLE_INTERVAL_SEC` (300 s) during simulation inside the C++ core and returned to Python via pybind11.

## Collected Metrics

| Metric | Definition |
|--------|------------|
| Wait variance | `Var(current_wait_sec)` across 34 rides (broken rides excluded when wait ≥ 9000) — **diagnostic only** for PPO |
| Mean wait | Average wait across rides at sample time |
| Rides completed | Count of `RideComplete` events |
| Rides per party | `rides_completed / total_parties` |
| Rides per guest | `rides_completed / total_guests` |
| Must-dos assigned / completed | Spawn-time must-do flags vs successful `RideComplete` clears |
| Must-do completion rate | `must_dos_completed / must_dos_assigned` |
| Preference score (guest-weighted) | Σ `preference[ride] × party_size` over completions; report per guest as `/ total_guests` |
| Must-do latency | Mean `(complete_sec − spawn_sec)` over must-do completions |
| Breakdown count | Number of breakdown events |
| Wall time | Elapsed time for `_park_sim.run_day()` |

PPO trains on preference / must-do latency rewards (not wait variance). See `docs/training.md`.

## Usage

```python
from simulator import run_day

metrics = run_day(seed=0)
print(metrics.rides_per_party, metrics.avg_wait_variance)
```
