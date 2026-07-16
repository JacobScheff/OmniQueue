# Native C++ Simulator

**Modules:** `native/`, `_park_sim` (pybind11 extension)

## Overview

The discrete event simulator and heuristic router run in **C++17** and are exposed to Python via **`_park_sim`**. PyTorch / RL code stays in Python; call `run_day(seed)` for fast rollouts.

## Build (Windows, macOS, Linux)

```bash
pip install -e .
```

Requires a C++17 compiler:

- **Windows:** Visual Studio Build Tools with "Desktop development with C++"
- **macOS:** Xcode command line tools
- **Linux:** `g++` or `clang++`

Regenerate static graph data after editing `config.py` or the park graph:

```bash
python tools/export_native_data.py
pip install -e .
```

## Python Usage

```python
from simulator import run_day, record_day, native_backend_name

print(native_backend_name())  # "native" or "unavailable"
metrics = run_day(seed=42)
metrics = run_day(seed=42, backend="native")

# Full-day event log for the Pygame visualizer (see docs/visualization.md)
recording = record_day(seed=42, sample_interval_sec=60)
```

Environment variable: `OMNIQUEUE_BACKEND=auto|native`

The legacy Python DES (`backend="python"`) was removed; build the native extension to run simulations.

## Benchmark

```bash
python benchmark.py --seed 42 --runs 5
python benchmark.py --seed 42 --runs 5 --backend native
```

Typical throughput is **~0.2s per full park day** (~50k guests) on modern hardware.

## RL / PyTorch Integration (Phase 2–3)

- `run_day(seed)` returns `DayMetrics` for throughput testing and behavioral cloning labels.
- `ParkEnv` exposes `reset` / `step` / `exchange_batch` for PPO rollouts (batched policy inference from Python).
- **Interactive / shadow play:** `ParkEnv.reset_play` + `play_advance` hybrid routing (focal human/heuristic/PPO, crowd heuristic or PPO), and `run_play_day` for heuristic crowd+focal with a custom focal profile. See `docs/interactive-play.md`.
- **Watch mid-day prefs:** `ParkEnv.play_update_focal_preferences` updates the focal guest’s preference weights / must-dos without resetting location or ride history; `play_focal_state` / `play_focal_ride_history` expose live focal status. See `docs/watch.md`.
- **Reward contract:** each routing step gets a **dense** wait-variance penalty (`-PPO_WAIT_VAR_STEP_COEF × var/1e6`), plus any **pending preference / must-do bonus** for the party being routed (earned at the previous `RideComplete`). Episode end adds `-avg_wait_variance/1000`, an unfulfilled must-do penalty, and any leftover pending preference. See `docs/training.md`.

## Notes

- C++ uses `std::mt19937_64` for party spawn and routing randomness.
- Operating day is `DAY_SECONDS` (54,000). Soft close continues the timing wheel for up to `CLOSE_DRAIN_SEC` afterward so queued/on-ride parties can finish and exit.
- Spawn/router/PPO reward constants in `native/include/park_sim.hpp` must stay in sync with `config.py` manually until a shared export step exists.
