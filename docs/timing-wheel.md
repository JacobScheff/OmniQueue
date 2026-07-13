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
- The wheel spans `DAY_SECONDS + CLOSE_DRAIN_SEC` so soft-close drain events (finish queue/ride, then exit) can schedule past official close.
- Seconds beyond the sim horizon are clamped to the horizon end.
- 64,801 bucket vectors are preallocated per simulation day (`15 h` operating day + `3 h` drain).
