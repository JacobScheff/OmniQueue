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

- **Today:** `run_day(seed)` returns `DayMetrics` for throughput testing and behavioral cloning labels.
- **Next:** extend `_park_sim` with `ParkEnv` step API and NumPy observation buffers for vectorized PPO.
- **Policy inference:** keep PyTorch model in Python; pass action tensors into C++ per routing batch.

## Notes

- C++ uses `std::mt19937_64` for party spawn and routing randomness.
- Spawn/router timing constants in `native/include/park_sim.hpp` must stay in sync with `config.py` manually until a shared export step exists.
- Heuristic routing only for now. PPO actions will be passed from Python in a later API.
