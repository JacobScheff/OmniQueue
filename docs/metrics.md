# Metrics

**Modules:** `metrics.py` (`DayMetrics`), `native/src/park_sim.cpp` (collection)

## Overview

KPIs are sampled every `METRICS_SAMPLE_INTERVAL_SEC` (300 s) during simulation inside the C++ core and returned to Python via pybind11.

## Collected Metrics

| Metric | Definition |
|--------|------------|
| Wait variance | `Var(current_wait_sec)` across 35 rides (broken rides excluded when wait ≥ 9000) |
| Mean wait | Average wait across rides at sample time |
| Rides completed | Count of `RideComplete` events |
| Rides per party | `rides_completed / total_parties` |
| Rides per guest | `rides_completed / total_guests` |
| Breakdown count | Number of breakdown events |
| Wall time | Elapsed time for `_park_sim.run_day()` |

PPO also shapes rewards with a **dense per-step wait-variance penalty** plus secondary preference / must-do terms. See `docs/training.md`.

## Usage

```python
from simulator import run_day

metrics = run_day(seed=0)
print(metrics.rides_per_party, metrics.avg_wait_variance)
```
