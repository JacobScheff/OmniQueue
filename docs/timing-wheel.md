# Timing Wheel

**Module:** `timing_wheel.py`

## Overview

The discrete event simulator uses a **bucket-array timing wheel**: one list per simulated second (0–54000). Scheduling appends to a bucket in **O(1)**; advancing time scans forward with a cursor until a non-empty bucket is found.

## API

| Method | Description |
|--------|-------------|
| `schedule(at_second, event)` | Append event to second `at_second` (O(1)) |
| `pop_next()` | Advance cursor to next non-empty second and return its events |
| `empty()` | True when no future events remain |
| `peek_time()` | Next occupied second without popping |

## Design Notes

- Same-second events preserve FIFO append order within the bucket.
- Seconds beyond `DAY_SECONDS` are clamped to the end of the day.
- A min-heap implementation remains a drop-in alternative if memory becomes a concern (54001 bucket lists).
