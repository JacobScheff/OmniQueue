# OmniQueue

AI-driven **personal next-ride planner** for theme-park parties. OmniQueue simulates a full Disneyland day (~50k guests, 34 rides, 8:00 AM–11:00 PM) and trains a policy that, for each party, picks the next ride given preferences, must-dos, walk times, and live waits.

The goal is to get preferred and must-do rides done quickly. Wait-time variance across rides is a diagnostic KPI, not the training objective.

All park code lives in [`Park/`](Park/).

The website can be viewed at: https://omniqueue.onrender.com

## How it works

1. **Discrete event simulator** — C++ DES at 1-second resolution, exposed to Python as `_park_sim`. Parties (not individuals) walk OSM pedestrian paths, join FIFO queues, ride, and re-route after completions or breakdowns. A full day rolls out in ~0.2s.
2. **Heuristic crowd** — preference-ordered balking with limited ride repeats. Used as the park background and as the expert for behavioral cloning.
3. **Personal planner (`rank_route_v1`)** — a pointer actor-critic that ranks candidate rides, then emits a short K-stop route. PPO trains N focal parties per day against the heuristic crowd; only `route[0]` is committed in the sim. Companion, Watch, and Play consume the same single-party policy.

Walking uses the committed Disneyland walkway network (`Park/data/pathways.json`) with near-shortest path sampling. Rides break down, queued parties evacuate one every 4 seconds, and the park soft-closes at 11:00 PM so already-queued / on-ride parties can finish.

## Setup

Requires Python 3.10+, a C++17 compiler (Visual Studio Build Tools on Windows, Xcode CLT on macOS, g++/clang++ on Linux), CMake, and PyTorch.

From `Park/`:

```bash
pip install -e .
python tools/export_native_data.py   # after changing config.py or the park graph
```

Confirm the native backend:

```bash
python -c "from simulator import native_backend_name; print(native_backend_name())"
```

That should print `native`. After editing `config.py` or `park_graph.py`, re-export graph data and reinstall — the C++ extension embeds walk times at compile time.

Tests (from `Park/`):

```bash
python -m pytest
```

## Run

All commands below assume cwd is `Park/`.

| What | Command |
|------|---------|
| Full-day sim + KPIs | `python benchmark.py --seed 42 --runs 3` |
| Replay a recorded day | `python visualize.py --seed 42` |
| Play as one guest vs the crowd | `python play.py --seed 42` |
| Watch a PPO focal through a day | `python watch.py --model checkpoints/ppo/ppo_final.pt` |
| Live Disneyland companion | `python run_companion.py` (API) + `cd companion/web && npm install && npm run dev` |

Visualization, Play, and Watch need Pygame: `pip install -e ".[viz]"`.

### Training

```bash
python training/bc_train.py --seed 42 --bc-days 1 --device cpu
python training/ppo_train.py --seed 42 --init-checkpoint checkpoints/bc/bc_final.pt --total-days 100 --num-focals 48 --device cpu
```

Hyperparameters live in `config.py` (`BC_*`, `PPO_*`). Play / Watch / visualize spawn popularity-weighted preferences; BC and PPO training use fully random prefs and must-dos so the policy has to read the preference vector.

### Companion (live waits)

Phone-first web app: pull real wait times, edit prefs / must-dos / location, and get a PPO route (ONNX). The DES is not used at request time.

```bash
pip install -r companion/requirements.txt
python run_companion.py
cd companion/web && npm install && npm run dev
```

See [`Park/docs/companion.md`](Park/docs/companion.md) for Docker and Render deploy.

## Layout

```
Park/
  config.py            rides, graph, spawn, router, training constants
  simulator.py         run_day() / record_day() → _park_sim
  park_graph.py        A* + precomputed walk matrix
  pathways.py          OSM walkway loader
  data/pathways.json   Disneyland pedestrian network
  native/              C++ DES, heuristic router, pybind11 bindings
  model.py             ParkRouterModel (rank-then-route actor-critic)
  training/            behavioral cloning + PPO
  visualize.py         Pygame park-day replay
  play.py              human-vs-AI interactive day
  watch.py             PPO-focal day watcher
  companion/           live wait-time companion (API + SPA)
  benchmark.py         throughput harness
  docs/                feature documentation
  tests/
```

## Docs

| Topic | Doc |
|-------|-----|
| Native simulator | [native-simulator.md](Park/docs/native-simulator.md) |
| Parties, spawn, prefs | [parties.md](Park/docs/parties.md) |
| Rides and queues | [rides-and-queues.md](Park/docs/rides-and-queues.md) |
| Heuristic router | [heuristic-router.md](Park/docs/heuristic-router.md) |
| Park graph / OSM walks | [park-graph.md](Park/docs/park-graph.md) |
| Rank-then-route model | [rank-route-architecture.md](Park/docs/rank-route-architecture.md) |
| BC + PPO | [training.md](Park/docs/training.md) |
| Visualization | [visualization.md](Park/docs/visualization.md) |
| Interactive play | [interactive-play.md](Park/docs/interactive-play.md) |
| Watch | [watch.md](Park/docs/watch.md) |
| Companion | [companion.md](Park/docs/companion.md) |
| Metrics | [metrics.md](Park/docs/metrics.md) |

Full index: [`Park/AGENTS.md`](Park/AGENTS.md).
