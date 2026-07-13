# Park Graph

**Module:** `park_graph.py`, `config.py`, `pathways.py`

## Overview

Walking times come from **near-shortest paths** on the **OSM pedestrian walkway network** (`data/pathways.json`), not straight-line ride-to-ride edges. Routing still uses a compact set of **macro nodes** (hubs + ride leaves) for the DES; each snaps to the nearest walkway node. Idle wandering still uses the hub topology in `MACRO_EDGES`.

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
| `pathways.py` | Loads JSON, near-shortest paths in meters, polylines for visualization |

Regenerate after OSM refreshes (requires network + `pip install osmnx`):

```bash
python tools/extract_osm_pathways.py   # also runs tools/simplify_pathways.py
python tools/export_native_data.py
pip install -e .
```

`simplify_pathways.py` applies layout polish: one straight Main Street, Pirates on
the Adventureland Y-junction, Haunted Mansion in the New Orleans path cluster,
Pooh on the Critter Country path bundle (HM↔Critter western link preserved),
Small World at the top of its plaza, Space Mountain / Indiana / Autopia stubs,
Critter↔GE western corridor retained (nowhere scribbles left of Rise removed),
and Buzz’s vertical spur.

Display coordinates in `config.RIDES` / `HUB_COORDS` are overwritten at import time from the pathways file when present.

## Near-shortest path randomization

Guests do **not** follow a live crowd-density map. Instead, when walking between two macro nodes, the DES samples among OSM paths whose length is within `WALK_PATH_LENGTH_SLACK` of the shortest (up to `WALK_PATH_MAX_VARIANTS` options):

```
P(path_i) ∝ exp(-(walk_sec_i − walk_sec_shortest) / WALK_PATH_SOFTMAX_TAU_SEC)
```

Similar lengths get similar probabilities; longer detours are rare. Heuristic routing / balking still uses the **shortest** walk time (`kBaseWalkToRides`); only the executed walk samples a variant. Set `WALK_PATH_RANDOM = False` for deterministic shortest-only walks.

| Config | Default | Meaning |
|--------|---------|---------|
| `WALK_PATH_RANDOM` | `True` | Enable length-weighted path sampling |
| `WALK_PATH_MAX_VARIANTS` | `6` | Cap alternatives per OD pair |
| `WALK_PATH_LENGTH_SLACK` | `0.15` | Allow paths ≤ 15% longer than shortest |
| `WALK_PATH_SOFTMAX_TAU_SEC` | `45` | Softmax temperature in walk-seconds |

Native export embeds `kWalkVariantCount` / `kWalkVariantBaseSec`. Each `WalkRecord` stores `path_variant` so visualization replays the same polyline.

## Precomputed Matrix

At startup / export, all-pairs shortest walk times between macro nodes are precomputed at nominal speed (`BASE_WALKING_SPEED` = 1.4 m/s), plus near-shortest variants. Runtime party walks scale by:

```
actual_sec = ceil(base_sec × BASE_WALKING_SPEED / party.effective_speed)
```

Python caches that matrix in **`cache/walk_matrix.npz`** (gitignored). A SHA-256 fingerprint of `data/pathways.json` plus walk-config knobs (`BASE_WALKING_SPEED`, `WALK_PATH_MAX_VARIANTS`, `WALK_PATH_LENGTH_SLACK`, node id set) decides whether the cache is still valid:

- Cache hit → `walk variants: loaded from cache` (fast).
- Miss / stale → recompute all OD pairs, then write the cache.

Force a rebuild with either:

```bash
rm -f cache/walk_matrix.npz
python -c "from park_graph import get_park_graph; get_park_graph(force_recompute=True)"
```

Native export still embeds the matrix into `graph_data.hpp` at compile time; the disk cache only speeds up Python (`visualize.py`, tests, `export_native_data.py`).

## Editing the Layout

1. Prefer regenerating `data/pathways.json` from OSM rather than hand-editing coordinates.
2. Adjust hub/ride fallback coordinates in `config.py` only when pathways data is absent.
3. Idle topology: add or remove edges in `MACRO_EDGES`; assign rides via `RIDE_HUB`.
4. Regenerate native data: `python tools/export_native_data.py` then `pip install -e .`
5. After pathways / walk-config changes, force-rebuild the Python walk cache (see above) if you need fresh variants before the next automatic miss.

## API

| Method | Description |
|--------|-------------|
| `walk_time(from_node, to_node)` | Shortest base walk seconds at nominal speed |
| `pathways.near_shortest_variants(...)` | Enumerate near-shortest OSM path options |
| `base_walk_to_rides` | `(num_nodes, NUM_RIDES)` shortest-time matrix exported to `graph_data.hpp` |
| `node_idx_to_ride` | Maps node index → ride id (-1 if not a ride) |
| `neighbors_within_hops(node, max_hops)` | Idle-wander candidates on macro adjacency |
| `path_polyline_for_idx(from_idx, to_idx, variant=0)` | Walkway polyline for a path variant (for viz) |
