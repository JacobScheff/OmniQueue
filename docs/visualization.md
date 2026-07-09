# Visualization (Phase 4)

**Modules:** `visualize.py`, `_park_sim.record_day`, `simulator.record_day`

## Overview

Pygame replays a full park day recorded from the C++ discrete event simulator. The DES runs once with the heuristic router, emits a compact event log, then the UI scrubs and animates parties on the macro park map.

## Run

```bash
pip install -e ".[viz]"   # or: pip install pygame
python visualize.py --seed 42
python visualize.py --seed 42 --speed 120 --sample-interval 60
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--seed` | `42` | RNG seed for the recorded day |
| `--speed` | `60` | Simulated seconds per real second |
| `--sample-interval` | `60` | Seconds between ride wait/status samples |
| `--max-seconds` | day length | Optional shorter replay window |

Controls: **Space** / Play-Pause, scrub slider, **+/-** speed, click a party in the sidebar to track.

## Recording API

```python
from simulator import record_day

rec = record_day(seed=42, sample_interval_sec=60)
# rec.parties, rec.walks, rec.ride_samples, rec.ride_completions, rec.metrics
```

Native entry point: `_park_sim.record_day(seed, sample_interval_sec)`.

### WalkRecord

| Field | Meaning |
|-------|---------|
| `party_id` | Party index |
| `start_sec` / `end_sec` | Actual walk interval (cancel shortens `end_sec`) |
| `planned_end_sec` | Scheduled arrival used for position interpolation |
| `from_idx` / `to_idx` | Graph node indices (same order as `ParkGraph`) |
| `target_ride` | Ride id, `-1` exit, or `-2` idle |
| `cancelled` | `1` if re-routed / interrupted before arrival |

### RideSample

Periodic snapshot: `wait[35]`, `broken[35]`, `queue_len[35]` at `sec`.

### PartyRideEvent

`party_id`, `sec`, `ride_id` for each credited ride completion (itinerary sidebar).

## Display Notes

- Walk paths are **straight lines between macro nodes** (hubs / rides), matching the graph the DES uses for travel times — not pathway polylines.
- Wait labels on rides show **minutes** (`wait_sec / 60`); broken rides show `X`.
- Crowd dots are subsampled when more than ~2500 parties are walking at once.
- PPO / trained-model routing is not wired into recording yet; visualization uses the built-in heuristic day.

## Dependencies

Optional extra: `pygame>=2.5` (`pip install -e ".[viz]"`).
