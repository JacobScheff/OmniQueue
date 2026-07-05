# Park Graph

**Module:** `park_graph.py`, `config.py`

## Overview

Walking times come from **A\*** pathfinding on a **macro hub-and-spoke graph**, not direct ride-to-ride edges. The graph is defined in `config.py` (`HUB_COORDS`, `MACRO_EDGES`, `RIDE_HUB`) and is easy to edit when real pathway data becomes available.

## Graph Layers

```
ENTRANCE → MAIN_HUB → {land hubs} → RIDE_* leaf nodes
```

Waypoint nodes (`NODE_RIVER_CROSSING`, `NODE_CENTRAL_PLAZA`) reduce unrealistic straight-line shortcuts.

## Precomputed Matrix

At startup, all-pairs walk times are precomputed at nominal speed (`BASE_WALKING_SPEED`) and stored in a NumPy array. Runtime party walks scale by:

```
actual_sec = ceil(base_sec × BASE_WALKING_SPEED / party.effective_speed)
```

## Editing the Layout

1. Adjust hub/ride coordinates in `config.py`.
2. Add or remove edges in `MACRO_EDGES`.
3. Assign rides to land hubs via `RIDE_HUB`.
4. Restart the simulator — the matrix rebuilds automatically.

## API

| Method | Description |
|--------|-------------|
| `walk_time(from_node, to_node)` | Base walk seconds at nominal speed |
| `party_walk_sec(from_idx, to_idx, speed)` | Walk seconds between node indices |
| `party_walk_to_ride_sec(from_idx, ride_id, speed)` | Walk seconds to a ride leaf |
| `base_walk_to_rides` | `(num_nodes, 35)` int32 matrix for Numba routing |
| `node_idx_to_ride` | Maps node index → ride id (-1 if not a ride) |
| `random_idle_node(rng, node, max_hops=2)` | Random reachable node for idle wandering |
