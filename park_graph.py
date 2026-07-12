"""Macro park graph, A* pathfinding, and precomputed walk-time matrix."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

import config
from pathways import load_pathways


@dataclass(frozen=True)
class Graph:
    node_coords: dict[int, tuple[float, float]]
    adjacency: dict[int, list[tuple[int, float]]]
    num_nodes: int
    walk_time_sec: list[list[int]]


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_graph() -> Graph:
    """Build hub-and-spoke macro graph with ride leaf nodes.

    When ``data/pathways.json`` is present, edge weights and all-pairs walk
    times use OSM pedestrian path lengths (meters). Otherwise Euclidean
    distances in display units are used (legacy fallback).
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

    walk_time = [[0] * n for _ in range(n)]
    speed = config.BASE_WALKING_SPEED

    if pathways is not None:
        # All-pairs along the real walkway network (not limited to MACRO_EDGES).
        for i, src in enumerate(node_ids):
            for j, dst in enumerate(node_ids):
                if i == j:
                    continue
                dist_m = pathways.path_length_m(src, dst)
                walk_time[i][j] = max(1, int(math.ceil(dist_m / speed)))
    else:
        for i, src in enumerate(node_ids):
            for j, dst in enumerate(node_ids):
                if i == j:
                    continue
                dist = astar_distance(adjacency, node_coords, src, dst)
                walk_time[i][j] = max(1, int(math.ceil(dist / speed)))

    return Graph(
        node_coords=node_coords,
        adjacency=adjacency,
        num_nodes=n,
        walk_time_sec=walk_time,
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

    def __init__(self) -> None:
        self._graph = build_graph()
        self._node_ids = sorted(self._graph.node_coords.keys())
        self._index_of = {nid: i for i, nid in enumerate(self._node_ids)}
        self.idx_to_node_id = np.array(self._node_ids, dtype=np.int32)
        self.num_nodes = len(self._node_ids)

        self.base_walk_matrix = np.array(self._graph.walk_time_sec, dtype=np.int32)
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
        self._path_polylines: dict[tuple[int, int], list[tuple[float, float]]] = {}

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

    def path_polyline_for_idx(
        self, from_idx: int, to_idx: int
    ) -> list[tuple[float, float]]:
        if from_idx == to_idx:
            nid = self.idx_to_node(from_idx)
            return [self._graph.node_coords[nid]]
        key = (from_idx, to_idx)
        cached = self._path_polylines.get(key)
        if cached is not None:
            return cached
        pathways = load_pathways()
        if pathways is not None:
            src = self.idx_to_node(from_idx)
            dst = self.idx_to_node(to_idx)
            poly = pathways.path_polyline(src, dst)
        else:
            a = self._graph.node_coords[self.idx_to_node(from_idx)]
            b = self._graph.node_coords[self.idx_to_node(to_idx)]
            poly = [a, b]
        self._path_polylines[key] = poly
        return poly


_GRAPH: ParkGraph | None = None
_COORDS_APPLIED = False


def get_park_graph() -> ParkGraph:
    global _GRAPH, _COORDS_APPLIED
    if not _COORDS_APPLIED:
        from pathways import apply_pathway_coords

        apply_pathway_coords(config)
        _COORDS_APPLIED = True
    if _GRAPH is None:
        _GRAPH = ParkGraph()
    return _GRAPH
