# Dynamic AI Theme Park Router — Agent Guide

## Project Goal

Build a centralized, AI-driven routing system for a theme park that dynamically directs **parties** (groups of guests) to their next attraction. The system optimizes:

1. **Wait variance** — balance load across all rides (primary KPI).
2. **Pathway congestion** — avoid bottlenecks (future phases).
3. **Guest satisfaction** — match parties to preferred rides.

Training uses a **Discrete Event Simulator (DES)** at 1-second resolution, implemented in **C++17** and exposed to Python via **`_park_sim`**. Phase 1 delivers fast rollouts and a built-in **heuristic router**. Later phases add PyTorch (Pointer Actor-Critic), PPO via CleanRL, and Pygame visualization.

## Enforced Rules

1. **Root-only code** — all implementation lives at the repo root or in subfolders (`router/`, `docs/`, `tests/`). Do not add new code under legacy reference folders.
2. **Never modify legacy folders** — treat them as read-only reference if needed; do not edit them.
3. **Party-based simulation** — route parties, not individuals. Party speed models min-of-member walking speeds.
4. **Second resolution** — park day is 8:00 AM–11:00 PM (54,000 seconds).
5. **Switchable router** — `config.ROUTER` selects `"heuristic"` (C++ built-in) or `"ppo"` (Phase 3 stub).
6. **Documentation stays current** — any behavior change must update the matching file in `docs/` in the same change.
7. **No land-themed preferences** — party preferences are random with must-do boosts only.
8. **Breakdown realism** — queued parties decide immediately at the ride entrance but evacuate one party every 4 seconds; on-ride parties evacuate last without ride completion credit.

## Repository Structure

```
/
  AGENTS.md           ← you are here
  config.py           ← rides, graph, spawn, router constants
  park_graph.py       ← A* pathfinding + precomputed walk matrix (export input)
  simulator.py        ← run_day() / record_day() Python wrappers → _park_sim
  visualize.py        ← Pygame park-day replay (Phase 4)
  metrics.py          ← DayMetrics dataclass
  model.py            ← ParkRouterModel (pointer actor-critic)
  training/           ← bc_train.py, ppo_train.py, eval_policy.py
  benchmark.py        ← performance harness
  native/             ← C++ DES core + heuristic router + pybind11 `_park_sim`
  tools/              ← export_native_data.py (graph → C++ header)
  router/
    base.py           ← Router protocol, get_router() (PPO Phase 3)
    ppo.py            ← Phase 3 stub
  docs/               ← feature documentation (see index below)
  tests/              ← unit and integration tests
```

## Documentation Index (`docs/`)

| Doc | Module | Description |
|-----|--------|-------------|
| [timing-wheel.md](docs/timing-wheel.md) | `native/src/park_sim.cpp` | Bucket-array scheduler, event batching |
| [park-graph.md](docs/park-graph.md) | `park_graph.py` | Macro graph, A*, walk matrix |
| [parties.md](docs/parties.md) | `native/src/park_sim.cpp` | Spawn, size, speed, preferences, must-do |
| [rides-and-queues.md](docs/rides-and-queues.md) | `native/src/park_sim.cpp` | Capacity, implicit FIFO boarding, wait calc |
| [breakdowns.md](docs/breakdowns.md) | `native/src/park_sim.cpp` | Breakdown, evacuation deque, re-route rules |
| [heuristic-router.md](docs/heuristic-router.md) | `native/src/park_sim.cpp` | Balking, idle walk, force-pick fallback |
| [metrics.md](docs/metrics.md) | `metrics.py`, C++ metrics | KPI definitions and sampling |
| [native-simulator.md](docs/native-simulator.md) | `native/`, `_park_sim` | C++ extension build and Python API |
| [benchmark.md](docs/benchmark.md) | `benchmark.py` | Performance harness |
| [training.md](docs/training.md) | `model.py`, `training/` | BC + PPO training and checkpoints |
| [visualization.md](docs/visualization.md) | `visualize.py`, `record_day` | Pygame park-day replay |

## Phase Roadmap

| Phase | Deliverable |
|-------|-------------|
| **1** | C++ DES + heuristic router + docs |
| **2** | PyTorch `ParkRouterModel` + behavioral cloning (`training/bc_train.py`) |
| **3** | PPO fine-tuning in `ParkEnv` (`training/ppo_train.py`) |
| **4** (current) | Pygame visualization of recorded heuristic days |

## Build

```bash
pip install -e .
python tools/export_native_data.py   # after config/graph changes
```

## Cursor Cloud specific instructions

- **System dependency:** building the `_park_sim` C++ extension requires the Python dev headers (`python3-dev`, providing `/usr/include/python3.12`). These are installed at the VM/snapshot level; without them `pip install -e .` fails at the CMake `find_package(pybind11)` step. This is not part of the per-startup update script.
- **Editable install uses `--user --break-system-packages`:** the base image marks the system Python as externally managed (PEP 668), so installs go to `~/.local`. Use `python3` / `python3 -m pytest` directly (there is no virtualenv to activate). The `pytest` console script lives in `~/.local/bin`, which is not on `PATH` — invoke it as `python3 -m pytest`.
- **After editing `config.py` or `park_graph.py`:** run `python tools/export_native_data.py` to regenerate `native/generated/graph_data.hpp`, then re-run `pip install -e .` — the C++ extension embeds that data at compile time and will not pick up changes otherwise.
- **Running things:** tests via `python3 -m pytest`; a full park-day simulation via `python3 benchmark.py --seed 42 --runs 3`; BC training via `python3 training/bc_train.py ...` (see `docs/training.md`); visualization via `python3 visualize.py --seed 42` (requires `pygame`, see `docs/visualization.md`).
- **Verify the native backend** is active with `python3 -c "from simulator import native_backend_name; print(native_backend_name())"` (should print `native`).
