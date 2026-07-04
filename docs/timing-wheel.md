# Timing Wheel

**Module:** `timing_wheel.py`

## Overview

The discrete event simulator uses a **min-heap** scheduler keyed by integer seconds. Events scheduled for the same second are batched and processed in FIFO enqueue order.

## API

| Method | Description |
|--------|-------------|
| `schedule(at_second, event)` | Insert event at `at_second` (O(log n)) |
| `pop_next()` | Advance to earliest second and return `(second, [events])` |
| `empty()` | True when no events remain |
| `peek_time()` | Next scheduled second without popping |

## Design Notes

- The heap stores `(at_second, sequence, Event)` tuples for stable ordering.
- A true O(1) timing wheel can replace this module later without changing event handlers.
- Park day runs from second `0` to `54000` (8:00 AM–11:00 PM).
