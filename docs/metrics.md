# Metrics

**Module:** `metrics.py`

## Overview

KPIs are sampled every `METRICS_SAMPLE_INTERVAL_SEC` (300 s) during simulation.

## Collected Metrics

| Metric | Definition |
|--------|------------|
| Wait variance | `Var(current_wait_sec)` across 35 rides (broken rides excluded from variance calc when wait ≥ 9000) |
| Mean wait | Average wait across rides at sample time |
| Rides completed | Count of `RIDE_COMPLETE` events |
| Rides per party | `rides_completed / total_parties` |
| Rides per guest | `rides_completed / total_guests` |
| Breakdown count | Number of breakdown events |
| Wall time | Python elapsed time for `run_day()` |

## Usage

```python
from simulator import run_day
metrics = run_day(seed=0)
print(metrics.avg_wait_variance)
print(metrics.rides_per_party)
```
