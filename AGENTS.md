# Dynamic AI Theme Park Router — Agent Guide

## Project Goal

Build a centralized, AI-driven routing system for a theme park that dynamically directs **parties** (groups of guests) to their next attraction. The system optimizes:

1. **Wait variance** — balance load across all rides (primary KPI).
2. **Pathway congestion** — avoid bottlenecks (future phases).
3. **Guest satisfaction** — match parties to preferred rides.

Training uses a **Discrete Event Simulator (DES)** with a **min-heap timing wheel** at 1-second resolution. Phase 1 delivers the fast simulator and a switchable **heuristic router**. Later phases add PyTorch (Pointer Actor-Critic), PPO via CleanRL, and Pygame visualization.

## Enforced Rules

1. **Root-only code** — all implementation lives at the repo root or in subfolders (`router/`, `docs/`, `tests/`). Do not add new code under legacy reference folders.
2. **Never modify legacy folders** — treat them as read-only reference if needed; do not edit them.
3. **Party-based simulation** — route parties, not individuals. Party speed models min-of-member walking speeds.
4. **Second resolution** — park day is 8:00 AM–11:00 PM (54,000 seconds).
5. **Switchable router** — `config.ROUTER` selects `"heuristic"` or `"ppo"`. Both must share the same observation/action interface for fair comparison.
6. **Documentation stays current** — any behavior change must update the matching file in `docs/` in the same change.
7. **No land-themed preferences** — party preferences are random with must-do boosts only.
8. **Breakdown realism** — queued parties decide immediately at the ride entrance but evacuate one party every 4 seconds; on-ride parties evacuate last without ride completion credit.

## Repository Structure

```
/
  AGENTS.md           ← you are here
  config.py           ← rides, graph, spawn, router constants
  park_types.py       ← EventType, PartyState, Event, Party, Ride
  timing_wheel.py     ← min-heap DES scheduler
  park_graph.py       ← A* pathfinding + precomputed walk matrix
  parties.py          ← party pool, spawn, preferences
  rides.py            ← ride state, queues, breakdown/evacuation
  events.py           ← event handler dispatch
  simulator.py        ← run_day() entry point
  metrics.py          ← KPI collection
  benchmark.py        ← performance harness
  router/
    base.py           ← Router protocol, get_router()
    heuristic.py      ← Phase 1 baseline router
    ppo.py            ← Phase 3 stub
  docs/               ← feature documentation (see index below)
  tests/              ← unit and integration tests
```

## Documentation Index (`docs/`)

| Doc | Module | Description |
|-----|--------|-------------|
| [timing-wheel.md](docs/timing-wheel.md) | `timing_wheel.py` | Min-heap scheduler, event batching |
| [park-graph.md](docs/park-graph.md) | `park_graph.py` | Macro graph, A*, walk matrix |
| [parties.md](docs/parties.md) | `parties.py` | Spawn, size, speed, preferences, must-do |
| [rides-and-queues.md](docs/rides-and-queues.md) | `rides.py` | Capacity, implicit FIFO boarding, wait calc |
| [breakdowns.md](docs/breakdowns.md) | `rides.py`, `events.py` | Breakdown, evacuation deque, re-route rules |
| [heuristic-router.md](docs/heuristic-router.md) | `router/heuristic.py` | Balking, idle walk, force-pick fallback |
| [metrics.md](docs/metrics.md) | `metrics.py` | KPI definitions and sampling |
| [benchmark.md](docs/benchmark.md) | `benchmark.py` | How to run perf tests |

## Phase Roadmap

| Phase | Deliverable |
|-------|-------------|
| **1** (current) | Fast DES + heuristic router + docs |
| **2** | PyTorch `ParkRouterModel` + behavioral cloning |
| **3** | CleanRL PPO integration |
| **4** | Pygame visualization with trained model |
