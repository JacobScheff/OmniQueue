# Timing Wheel

**Module:** `native/src/park_sim.cpp` (`TimingWheel` class)

## Overview

The discrete event simulator uses a **bucket-array timing wheel**: one list per simulated second (0–54000). Scheduling appends to a bucket in **O(1)**; advancing time scans forward with a cursor until a non-empty bucket is found.

## API

| Method | Description |
|--------|-------------|
| `schedule(at_second, event)` | Append event to second `at_second` (O(1)) |
| `pop_next()` | Advance cursor to next non-empty second and return its events |
| `empty()` | True when no future events remain |

## Design Notes

- Same-second events preserve FIFO append order within the bucket.
- Seconds beyond `DAY_SECONDS` are clamped to the end of the day.
- 54,001 bucket vectors are preallocated per simulation day.
