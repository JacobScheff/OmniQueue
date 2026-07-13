"""Macro park graph, A* pathfinding, and precomputed walk-time matrix."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from pathways import PATHWAYS_PATH, load_pathways

CACHE_DIR = Path(__file__).resolve().parent / "cache"
WALK_MATRIX_CACHE_PATH = CACHE_DIR / "walk_matrix.npz"
WALK_POLYLINES_CACHE_PATH = CACHE_DIR / "walk_polylines.npz"


@dataclass(frozen=True)
class Graph:
    node_coords: dict[int, tuple[float, float]]
    adjacency: dict[int, list[tuple[int, float]]]
    num_nodes: int
    walk_time_sec: list[list[int]]
    # walk_variant_count[i][j] in 1..WALK_PATH_MAX_VARIANTS
    walk_variant_count: list[list[int]]
    # walk_variant_base_sec[i][j][k]; unused slots are 0
    walk_variant_base_sec: list[list[list[int]]]


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _walk_matrix_fingerprint(node_ids: list[int]) -> str:
    """Hash pathways + config knobs that affect near-shortest walk times."""
    h = hashlib.sha256()
    if PATHWAYS_PATH.is_file():
        h.update(PATHWAYS_PATH.read_bytes())
    else:
        h.update(b"no-pathways")
    payload = {
        "node_ids": list(node_ids),
        "base_walking_speed": float(config.BASE_WALKING_SPEED),
        "max_variants": int(config.WALK_PATH_MAX_VARIANTS),
        "length_slack": float(config.WALK_PATH_LENGTH_SLACK),
    }
    h.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def _load_walk_matrix_cache(
    fingerprint: str, num_nodes: int, k_max: int
) -> tuple[list[list[int]], list[list[int]], list[list[list[int]]]] | None:
    if not WALK_MATRIX_CACHE_PATH.is_file():
        return None
    try:
        data = np.load(WALK_MATRIX_CACHE_PATH, allow_pickle=False)
        if str(data["fingerprint"]) != fingerprint:
            return None
        walk_time = data["walk_time_sec"]
        variant_count = data["walk_variant_count"]
        variant_base = data["walk_variant_base_sec"]
        if walk_time.shape != (num_nodes, num_nodes):
            return None
        if variant_count.shape != (num_nodes, num_nodes):
            return None
        if variant_base.shape != (num_nodes, num_nodes, k_max):
            return None
        return (
            walk_time.astype(np.int32).tolist(),
            variant_count.astype(np.int32).tolist(),
            variant_base.astype(np.int32).tolist(),
        )
    except (OSError, KeyError, ValueError):
        return None


def _save_walk_matrix_cache(
    fingerprint: str,
    walk_time: list[list[int]],
    variant_count: list[list[int]],
    variant_base: list[list[list[int]]],
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        WALK_MATRIX_CACHE_PATH,
        fingerprint=np.asarray(fingerprint),
        walk_time_sec=np.asarray(walk_time, dtype=np.int32),
        walk_variant_count=np.asarray(variant_count, dtype=np.uint8),
        walk_variant_base_sec=np.asarray(variant_base, dtype=np.int32),
    )


def _load_polyline_cache(
    fingerprint: str,
) -> dict[tuple[int, int, int], tuple[list[tuple[float, float]], list[float], float]] | None:
    if not WALK_POLYLINES_CACHE_PATH.is_file():
        return None
    try:
        data = np.load(WALK_POLYLINES_CACHE_PATH, allow_pickle=False)
        if str(data["fingerprint"]) != fingerprint:
            return None
        keys = np.asarray(data["keys"], dtype=np.int32)
        offsets = np.asarray(data["offsets"], dtype=np.int32)
        coords = np.asarray(data["coords"], dtype=np.float32)
        if keys.ndim != 2 or keys.shape[1] != 3:
            return None
        if len(offsets) != len(keys) + 1:
            return None
        from pathways import polyline_arc_lengths

        out: dict[
            tuple[int, int, int], tuple[list[tuple[float, float]], list[float], float]
        ] = {}
        for i, (a, b, v) in enumerate(keys):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            poly = [(float(x), float(y)) for x, y in coords[start:end]]
            cum, total = polyline_arc_lengths(poly)
            out[(int(a), int(b), int(v))] = (poly, cum, total)
        return out
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _save_polyline_cache(
    fingerprint: str,
    polylines: dict[
        tuple[int, int, int], tuple[list[tuple[float, float]], list[float], float]
    ],
) -> None:
    if not polylines:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keys = np.zeros((len(polylines), 3), dtype=np.int32)
    offsets = np.zeros(len(polylines) + 1, dtype=np.int32)
    chunks: list[np.ndarray] = []
    cursor = 0
    for i, ((a, b, v), (poly, _cum, _total)) in enumerate(sorted(polylines.items())):
        keys[i] = (a, b, v)
        offsets[i] = cursor
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        chunks.append(arr)
        cursor += len(arr)
    offsets[-1] = cursor
    coords = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), dtype=np.float32)
    np.savez_compressed(
        WALK_POLYLINES_CACHE_PATH,
        fingerprint=np.asarray(fingerprint),
        keys=keys,
        offsets=offsets,
        coords=coords,
    )


def build_graph(*, force_recompute: bool = False) -> Graph:
    """Build hub-and-spoke macro graph with ride leaf nodes.

    When ``data/pathways.json`` is present, edge weights and all-pairs walk
    times use OSM pedestrian path lengths (meters). Otherwise Euclidean
    distances in display units are used (legacy fallback).

    Near-shortest walk matrices are loaded from ``cache/walk_matrix.npz`` when
    the fingerprint still matches; pass ``force_recompute=True`` (or delete the
    cache file) to rebuild.
    """
    pathways = load_pathways()
    node_coords: dict[int, tuple[float, float]] = dict(config.HUB_COORDS)

    for ride_id, ride in enumerate(config.RIDES):
        node_coords[config.ride_node_id(ride_id)] = ride["coords"]

    adjacency: dict[int, list[tuple[int, float]]] = {n: [] for n in node_coords}

    def edge_weight(a: int, b: int) -> float:
        if pathways is not None:
            return pathways.path_length_m(a, b)
        return _euclidean(node_coords[a], node_coords[b])

    def add_edge(a: int, b: int) -> None:
        dist = edge_weight(a, b)
        adjacency[a].append((b, dist))
        adjacency[b].append((a, dist))

    for a, b in config.MACRO_EDGES:
        add_edge(a, b)

    for ride_id, ride in enumerate(config.RIDES):
        hub = config.RIDE_HUB[ride_id]
        add_edge(hub, config.ride_node_id(ride_id))

    node_ids = sorted(node_coords.keys())
    n = len(node_ids)
    k_max = max(1, int(config.WALK_PATH_MAX_VARIANTS))
    speed = config.BASE_WALKING_SPEED

    walk_time = [[0] * n for _ in range(n)]
    variant_count = [[1] * n for _ in range(n)]
    variant_base = [[[0] * k_max for _ in range(n)] for _ in range(n)]

    if pathways is not None:
        fingerprint = _walk_matrix_fingerprint(node_ids)
        cached = None if force_recompute else _load_walk_matrix_cache(fingerprint, n, k_max)
        if cached is not None:
            print("  walk variants: loaded from cache", flush=True)
            walk_time, variant_count, variant_base = cached
        else:
            # All-pairs along the real walkway network (not limited to MACRO_EDGES).
            total = n * (n - 1)
            done = 0
            for i, src in enumerate(node_ids):
                for j, dst in enumerate(node_ids):
                    if i == j:
                        continue
                    variants = pathways.near_shortest_variants(src, dst)
                    secs = [
                        max(1, int(math.ceil(v.length_m / speed))) for v in variants
                    ]
                    variant_count[i][j] = len(secs)
                    for k, sec in enumerate(secs):
                        variant_base[i][j][k] = sec
                    walk_time[i][j] = secs[0]
                    done += 1
                    if done % 200 == 0:
                        print(f"  walk variants: {done}/{total}", flush=True)
            _save_walk_matrix_cache(fingerprint, walk_time, variant_count, variant_base)
            print(f"  walk variants: cached → {WALK_MATRIX_CACHE_PATH}", flush=True)
    else:
        for i, src in enumerate(node_ids):
            for j, dst in enumerate(node_ids):
                if i == j:
                    continue
                dist = astar_distance(adjacency, node_coords, src, dst)
                sec = max(1, int(math.ceil(dist / speed)))
                walk_time[i][j] = sec
                variant_count[i][j] = 1
                variant_base[i][j][0] = sec

    return Graph(
        node_coords=node_coords,
        adjacency=adjacency,
        num_nodes=n,
        walk_time_sec=walk_time,
        walk_variant_count=variant_count,
        walk_variant_base_sec=variant_base,
    )


def astar_distance(
    adjacency: dict[int, list[tuple[int, float]]],
    coords: dict[int, tuple[float, float]],
    start: int,
    goal: int,
) -> float:
    if start == goal:
        return 0.0

    open_heap: list[tuple[float, int]] = [(0.0, start)]
    g_score: dict[int, float] = {start: 0.0}
    goal_coord = coords[goal]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return g_score[current]

        for neighbor, edge_cost in adjacency[current]:
            tentative = g_score[current] + edge_cost
            if tentative < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative
                h = _euclidean(coords[neighbor], goal_coord)
                heapq.heappush(open_heap, (tentative + h, neighbor))

    return _euclidean(coords[start], coords[goal])


class ParkGraph:
    """Runtime wrapper with node-id mapping and precomputed walk caches."""

    def __init__(self, *, force_recompute: bool = False) -> None:
        self._graph = build_graph(force_recompute=force_recompute)
        self._node_ids = sorted(self._graph.node_coords.keys())
        self._index_of = {nid: i for i, nid in enumerate(self._node_ids)}
        self.idx_to_node_id = np.array(self._node_ids, dtype=np.int32)
        self.num_nodes = len(self._node_ids)
        self._walk_fingerprint = _walk_matrix_fingerprint(self._node_ids)

        self.base_walk_matrix = np.array(self._graph.walk_time_sec, dtype=np.int32)
        self.walk_variant_count = np.array(self._graph.walk_variant_count, dtype=np.uint8)
        self.walk_variant_base_sec = np.array(
            self._graph.walk_variant_base_sec, dtype=np.int32
        )
        self._ride_col_indices = np.array(
            [self._index_of[config.ride_node_id(r)] for r in range(config.NUM_RIDES)],
            dtype=np.int32,
        )
        self.base_walk_to_rides = self.base_walk_matrix[:, self._ride_col_indices].copy()

        self.entrance_node_idx = np.int32(self._index_of[config.NODE_ENTRANCE])
        self.node_idx_to_ride = np.full(self.num_nodes, -1, dtype=np.int16)
        for ride_id in range(config.NUM_RIDES):
            idx = self._index_of[config.ride_node_id(ride_id)]
            self.node_idx_to_ride[idx] = ride_id

        self.ride_node_idx = np.array(
            [self._index_of[config.ride_node_id(r)] for r in range(config.NUM_RIDES)],
            dtype=np.int32,
        )

        # Lazy cache of walkway polylines between routing node indices (visualization).
        # Value: (points, cumulative arc lengths, total length)
        self._path_polylines: dict[
            tuple[int, int, int], tuple[list[tuple[float, float]], list[float], float]
        ] = {}
        self._polyline_cache_loaded = False
        self.load_polyline_cache()

    @property
    def entrance_node(self) -> int:
        return config.NODE_ENTRANCE

    def node_to_idx(self, node_id: int) -> int:
        return self._index_of[node_id]

    def idx_to_node(self, node_idx: int) -> int:
        return int(self.idx_to_node_id[node_idx])

    def ride_node(self, ride_id: int) -> int:
        return config.ride_node_id(ride_id)

    def walk_time(self, from_node: int, to_node: int) -> int:
        i = self._index_of[from_node]
        j = self._index_of[to_node]
        return int(self.base_walk_matrix[i, j])

    def walk_times_to_rides(self, from_node: int, effective_speed: float) -> np.ndarray:
        row_idx = self._index_of[from_node]
        scale = config.BASE_WALKING_SPEED / max(effective_speed, 0.1)
        cols = self.base_walk_to_rides[row_idx]
        return np.maximum(1, np.ceil(cols * scale)).astype(np.int32)

    def neighbors_within_hops(self, start_node: int, max_hops: int) -> list[int]:
        visited = {start_node}
        frontier = [start_node]
        results: list[int] = []
        for _ in range(max_hops):
            next_frontier: list[int] = []
            for node in frontier:
                for neighbor, _ in self._graph.adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        results.append(neighbor)
            frontier = next_frontier
        return results

    def load_polyline_cache(self) -> int:
        """Load packed walk polylines from disk. Returns number of entries loaded."""
        if self._polyline_cache_loaded:
            return len(self._path_polylines)
        cached = _load_polyline_cache(self._walk_fingerprint)
        self._polyline_cache_loaded = True
        if not cached:
            return 0
        self._path_polylines.update(cached)
        return len(cached)

    def save_polyline_cache(self) -> None:
        """Persist the in-memory walk polyline cache to disk."""
        _save_polyline_cache(self._walk_fingerprint, self._path_polylines)

    def path_polyline_for_idx(
        self, from_idx: int, to_idx: int, variant: int = 0
    ) -> list[tuple[float, float]]:
        return self.path_arc_for_idx(from_idx, to_idx, variant=variant)[0]

    def path_arc_for_idx(
        self, from_idx: int, to_idx: int, variant: int = 0
    ) -> tuple[list[tuple[float, float]], list[float], float]:
        """Return (polyline, cumulative arc lengths, total length) for a walk."""
        from pathways import polyline_arc_lengths

        if from_idx == to_idx:
            nid = self.idx_to_node(from_idx)
            poly = [self._graph.node_coords[nid]]
            return poly, [0.0], 0.0
        key = (from_idx, to_idx, int(variant))
        cached = self._path_polylines.get(key)
        if cached is not None:
            return cached
        pathways = load_pathways()
        if pathways is not None:
            src = self.idx_to_node(from_idx)
            dst = self.idx_to_node(to_idx)
            # Populate every near-shortest variant for this OD while we pay for enumeration.
            variants = pathways.near_shortest_variants(src, dst)
            for k, path in enumerate(variants):
                poly_k = list(path.polyline)
                cum_k, total_k = polyline_arc_lengths(poly_k)
                self._path_polylines[(from_idx, to_idx, k)] = (poly_k, cum_k, total_k)
            idx = max(0, min(int(variant), len(variants) - 1))
            return self._path_polylines[(from_idx, to_idx, idx)]
        a = self._graph.node_coords[self.idx_to_node(from_idx)]
        b = self._graph.node_coords[self.idx_to_node(to_idx)]
        poly = [a, b]
        cum, total = polyline_arc_lengths(poly)
        packed = (poly, cum, total)
        self._path_polylines[key] = packed
        return packed


_GRAPH: ParkGraph | None = None
_COORDS_APPLIED = False


def reset_park_graph() -> None:
    """Drop the process-local singleton (tests / forced rebuild)."""
    global _GRAPH
    _GRAPH = None


def get_park_graph(*, force_recompute: bool = False) -> ParkGraph:
    global _GRAPH, _COORDS_APPLIED
    if not _COORDS_APPLIED:
        from pathways import apply_pathway_coords

        apply_pathway_coords(config)
        _COORDS_APPLIED = True
    if force_recompute:
        _GRAPH = None
    if _GRAPH is None:
        _GRAPH = ParkGraph(force_recompute=force_recompute)
    return _GRAPH
