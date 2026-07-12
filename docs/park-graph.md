# Park Graph

**Module:** `park_graph.py`, `config.py`, `pathways.py`

## Overview

Walking times come from **A\*** / shortest-path distances on the **OSM pedestrian walkway network** (`data/pathways.json`), not straight-line ride-to-ride edges. Routing still uses a compact set of **macro nodes** (hubs + ride leaves) for the DES; each snaps to the nearest walkway node. Idle wandering still uses the hub topology in `MACRO_EDGES`.

Without `data/pathways.json`, the graph falls back to Euclidean distances on the macro hub-and-spoke layout.

## Graph Layers

```
ENTRANCE → MAIN_HUB → {land hubs} → RIDE_* leaf nodes
         ↘ walkway network (OSM) supplies meters + polylines
```

Waypoint nodes (`NODE_RIVER_CROSSING`, `NODE_CENTRAL_PLAZA`) still participate in idle-hop adjacency.

## OSM Pathways

| Artifact | Role |
|----------|------|
| `tools/extract_osm_pathways.py` | Downloads Disneyland walkways via **osmnx**, matches ride POIs, projects to display coords, writes JSON |
| `data/pathways.json` | Committed walkway nodes/edges (length in meters + simplified polylines) + snapped ride/hub positions |
| `pathways.py` | Loads JSON, shortest paths in meters, polylines for visualization |

Regenerate after OSM refreshes (requires network + `pip install osmnx`):

```bash
python tools/extract_osm_pathways.py   # also runs tools/simplify_pathways.py
python tools/export_native_data.py
pip install -e .
```

`simplify_pathways.py` applies a **minimal** pass: ride nudges (Indiana beside Jungle
Cruise; Rise northwest on Galaxy's Edge), Buzz vertical spur, a tiny eastern plaza
trim, and dangling-leaf cleanup. Critter↔Galaxy's Edge and Matterhorn↔Tomorrowland
corridors are preserved with most OSM walkway detail intact.

Display coordinates in `config.RIDES` / `HUB_COORDS` are overwritten at import time from the pathways file when present.

## Precomputed Matrix

At startup, all-pairs walk times between macro nodes are precomputed at nominal speed (`BASE_WALKING_SPEED` = 1.4 m/s) and stored in a NumPy array. Runtime party walks scale by:

```
actual_sec = ceil(base_sec × BASE_WALKING_SPEED / party.effective_speed)
```

## Editing the Layout

1. Prefer regenerating `data/pathways.json` from OSM rather than hand-editing coordinates.
2. Adjust hub/ride fallback coordinates in `config.py` only when pathways data is absent.
3. Idle topology: add or remove edges in `MACRO_EDGES`; assign rides via `RIDE_HUB`.
4. Regenerate native data: `python tools/export_native_data.py` then `pip install -e .`

## API

| Method | Description |
|--------|-------------|
| `walk_time(from_node, to_node)` | Base walk seconds at nominal speed |
| `party_walk_sec(from_idx, to_idx, speed)` | Walk seconds between node indices |
| `party_walk_to_ride_sec(from_idx, ride_id, speed)` | Walk seconds to a ride leaf |
| `base_walk_to_rides` | `(num_nodes, 35)` int32 matrix exported to `graph_data.hpp` |
| `node_idx_to_ride` | Maps node index → ride id (-1 if not a ride) |
| `neighbors_within_hops(node, max_hops)` | Idle-wander candidates on macro adjacency |
| `path_polyline_for_idx(from_idx, to_idx)` | Walkway polyline in display coords (for viz) |
