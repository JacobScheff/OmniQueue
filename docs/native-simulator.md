# Native C++ Simulator

**Modules:** `native/`, `_park_sim` (pybind11 extension)

## Overview

The performance-critical discrete event simulator runs in **C++17** and is exposed to Python via **`_park_sim`**. PyTorch / RL code stays in Python; call `run_day(seed)` for fast rollouts.

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
from simulator import run_day, native_backend_name

print(native_backend_name())  # "native" or "unavailable"
metrics = run_day(seed=42)            # auto: C++ if built
metrics = run_day(seed=42, backend="native")
metrics = run_day(seed=42, backend="python")  # reference impl
```

Environment variable: `OMNIQUEUE_BACKEND=auto|native|python`

## Benchmark

```bash
python benchmark.py --seed 42 --runs 5 --backend auto
python benchmark.py --seed 42 --runs 5 --backend native
```

On a typical Linux VM, native runs are **~25–30× faster** than the Python+Numba reference (~0.2s vs ~6.5s per day). Windows without Numba is often ~24s/day in Python; native should land near **sub-second** once built with `pip install -e .`.

## RL / PyTorch Integration (Phase 2–3)

- **Today:** `run_day(seed)` returns `DayMetrics` for throughput testing and behavioral cloning labels.
- **Next:** extend `_park_sim` with `ParkEnv` step API and NumPy observation buffers for vectorized PPO.
- **Policy inference:** keep PyTorch model in Python; pass action tensors into C++ per routing batch.

## Notes

- C++ uses `std::mt19937_64`; metrics will **not** match the Python reference byte-for-byte for the same seed.
- C++ backend supports **heuristic routing only** for now. PPO actions will be passed from Python in a later API.
