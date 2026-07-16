# Fleet Dispatch — Implementation Plan

Plan only. No code in this folder yet.

Sibling project to the theme-park router: reuse the same **pointer actor-critic + DES + BC/PPO** pattern for **EV / ride-hail style vehicle dispatch** in a city graph.

---

## 1. Goal

Train a centralized policy that assigns free vehicles to:

1. **Pick up** a pending guest request (origin → destination).
2. **Charge** at a charging station.
3. **Idle / reposition** (optional) or **STAY**.

Optimize contested resources under congestion (guest wait, fleet utilization, SOC feasibility) — analogous to park wait-variance + preference rewards.

---

## 2. Problem framing

| Park router | Fleet dispatch |
|-------------|----------------|
| Parties | Vehicles (decision agents) |
| Rides | Typed candidates: requests, chargers, idle anchors |
| Walk graph (OSM pathways) | Street graph (nodes = intersections, edges = streets) |
| Heuristic → BC → PPO | Same training stack |
| C++ DES (`_park_sim`) | New C++ DES (e.g. `_fleet_sim`) + Python training |

The street graph is for **simulation and ETA/energy**, not the raw policy action space.

---

## 3. Architecture summary

### 3.1 Simulator (mesoscopic DES)

- Event-driven day (or shift) loop — not full car-following physics in the RL hot path.
- Vehicles: location, SOC, capacity, status (idle / en route / serving / charging).
- Requests: origin, destination, spawn time, wait time, size.
- Chargers: node (or facility id), queue, power, compatibility.
- Travel: A* or cached shortest paths on the street graph; optional time-dependent edge speeds for traffic.
- On policy assign → enqueue drive / serve / charge events; on completion → next decision.

**Language:** C++17 core + pybind11; Python for config, training, eval, viz.  
Pure Python only for early API prototyping — not for PPO-scale rollouts.

### 3.2 Policy (pointer actor-critic)

Decision unit: one free vehicle (optionally a **wave** of free vehicles with a small coordinator transformer, park-style).

**Action = masked pointer over a dynamic candidate set**, not a fixed `3 × N` node grid:

| Candidate type | Selects | Typical mask |
|----------------|---------|--------------|
| `PICKUP` | a **request** (not merely a node) | taken, unreachable, SOC can’t finish trip (+ reserve), capacity |
| `CHARGE` | a **charger** | incompatible, unreachable, SOC not needed (optional soft) |
| `IDLE` | a small set of **anchors** (zones / hotspots), not all intersections | unreachable |
| `STAY` | special logit | rarely masked |

Sampling: one `Categorical` over masked logits (optional hierarchical `P(type)` × `P(target|type)` is equivalent if built from the same candidates).

### 3.3 Observations

Structured tensors (variable counts + padding masks), not a dense `k × (G+V) × N` grid:

- **Vehicle:** SOC, capacity, location encoding, time-to-free, etc.
- **Requests:** o/d encodings, wait time, size, trip length, deadline.
- **Chargers:** location, queue, power, ETA from vehicle.
- **Idle anchors:** local demand proxies, distance/ETA.
- **Env:** time-of-day, fleet-wide SOC / backlog stats.

**Pairwise (vehicle ↔ candidate):** drive time, distance, energy-to-reach (and for pickup: energy for o→d and optional post-trip charge). Computed from the graph + traffic each decision — not learned only from IDs.

### 3.4 Learned vs dynamic features

| Learned (`nn.Embedding`) | Dynamic (MLP inputs) |
|--------------------------|----------------------|
| Zone / charger / vehicle-type ids (stable catalogs) | Wait, SOC, queues, open/closed, traffic ETAs |
| Optional hour-of-day embedding | Pairwise eta / dist / energy |

Combine: `h = Norm(embed(id) + MLP(dynamic + pairwise))`, then `score = q·k/√d` (+ optional travel-time edge bias). Avoid `Embedding(N×N)` for OD pairs.

---

## 4. Why not `3 × N` over intersections

- Pickup needs a **request identity** when multiple guests share a node.
- City `N` is large; most nodes are irrelevant each step.
- Idle vs charge share “go to place” geometry; differ by **candidate set + mask**.
- Three independent row-softmaxes still need a type choice; one masked pointer is simpler and matches `ParkRouterModel`.

---

## 5. City scale for training

Size by **fleet contention** and **decisions per episode**, not map area alone.

| Stage | Graph | Fleet | Purpose |
|-------|-------|-------|---------|
| Debug | ~50–200 nodes, few chargers | 10–30 | shapes, masks, correctness |
| Learn | ~500–2k nodes or zone graph | 100–500 | BC + early PPO |
| Serious | districts / zones for policy; detailed streets for routing | 500–2k+ | real KPIs |

Targets:

- Enough demand that **queues form** (otherwise nothing to learn).
- On the order of **50k–200k+** assignment decisions per long episode (or many shorter episodes), similar spirit to park routing volume.
- Policy candidates stay small (`requests + chargers + anchors`); pathfinding graph may be larger.

---

## 6. External libraries and data

| Use | Tool |
|-----|------|
| Street network | OSM (OSMnx or export pipeline like park `pathways`) |
| Optional traffic calibration | SUMO (offline / periodic speeds) — avoid full SUMO-in-the-loop for every PPO step unless necessary |
| Large ABM reference | MATSim (research; not the default RL env) |
| Training | PyTorch; BC then PPO (CleanRL-style or park `training/` patterns) |

No off-the-shelf library provides pointer dispatch + EV SOC + PPO end-to-end. Plan: **custom DES + OSM graph**; SUMO optional for travel-time fidelity.

---

## 7. Proposed repo layout (future)

All new implementation under `fleet/` (this folder). Do not put fleet code under legacy reference trees.

```
fleet/
  PLAN.md              ← this document
  README.md            ← usage (later)
  config.py            ← graph, fleet, demand, train hparams (later)
  docs/                ← feature docs mirroring park docs style (later)
  native/              ← C++ DES + pybind `_fleet_sim` (later)
  model.py             ← pointer actor-critic (later)
  training/            ← bc_train, ppo_train, eval (later)
  tools/               ← OSM/graph export (later)
  tests/               ← unit / integration (later)
```

Exact filenames can adjust during implementation; keep the park rule: **behavior changes update matching docs in the same change**.

---

## 8. Phased roadmap

| Phase | Deliverable |
|-------|-------------|
| **0** | This plan; freeze action/obs contract on paper |
| **1** | Graph load (OSM or synthetic), mesoscopic DES stub, heuristic dispatcher |
| **2** | Python obs/action adapters + masking; toy BC on heuristic labels |
| **3** | C++ DES + pybind; performance parity goals vs park-style rollouts |
| **4** | Pointer model + BC at learn-scale (100–500 vehicles) |
| **5** | PPO (wait / SOC / coverage rewards); eval harness |
| **6** | Optional: SUMO-calibrated speeds, viz, multi-district graphs |

---

## 9. Reward sketch (PPO)

Primary: reduce **guest wait** (mean / variance / tail).  
Secondary: SOC feasibility (avoid stranded vehicles), charger congestion, optional coverage / idle miles penalty.  
Preference-style terms only if requests have priorities or SLAs.

Emit rewards on **assignment decisions** (and terminal bonuses), analogous to park routing-step rewards.

---

## 10. Open decisions (resolve before Phase 1 code)

1. Episode length: full day vs peak-hour shifts.
2. Whether Phase 1 includes **IDLE anchors** or only pickup / charge / STAY.
3. Zone graph vs full intersection graph for training.
4. Single passenger vs multi-capacity / pooling.
5. How strongly traffic feedback updates edge speeds online vs periodic refresh.

---

## 11. Relationship to the park project

- **Reuse ideas:** pointer head, action masks, wave coordinator cap, BC→PPO, C++ DES + Python training.
- **Do not:** conflate configs, native extensions, or checkpoints with the park simulator.
- Shared inspiration only; `fleet/` stays a separate product surface under the repo root.
