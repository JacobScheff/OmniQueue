"""Macro park graph, A* pathfinding, and precomputed walk-time matrix."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class Graph:
    node_coords: dict[int, tuple[float, float]]
    adjacency: dict[int, list[tuple[int, float]]]
    num_nodes: int
    walk_time_sec: list[list[int]]


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_graph() -> Graph:
    """Build hub-and-spoke macro graph with ride leaf nodes."""
    node_coords: dict[int, tuple[float, float]] = dict(config.HUB_COORDS)

    for ride_id, ride in enumerate(config.RIDES):
        node_coords[config.ride_node_id(ride_id)] = ride["coords"]

    adjacency: dict[int, list[tuple[int, float]]] = {n: [] for n in node_coords}

    def add_edge(a: int, b: int) -> None:
        ca, cb = node_coords[a], node_coords[b]
        dist = _euclidean(ca, cb)
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

    for i, src in enumerate(node_ids):
        for j, dst in enumerate(node_ids):
            if i == j:
                continue
            dist = astar_distance(adjacency, node_coords, src, dst)
            speed = config.BASE_WALKING_SPEED
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

        self._ride_node_to_id = {
            config.ride_node_id(r): r for r in range(config.NUM_RIDES)
        }
        self._walk_array = np.array(self._graph.walk_time_sec, dtype=np.float32)
        self.base_walk_matrix = np.array(self._graph.walk_time_sec, dtype=np.int32)
        self._ride_col_indices = np.array(
            [self._index_of[config.ride_node_id(r)] for r in range(config.NUM_RIDES)],
            dtype=np.int32,
        )
        self.base_walk_to_rides = self.base_walk_matrix[:, self._ride_col_indices].copy()
        self._speed_scale = config.BASE_WALKING_SPEED

        self.entrance_node_idx = np.int32(self._index_of[config.NODE_ENTRANCE])
        self.node_idx_to_ride = np.full(self.num_nodes, -1, dtype=np.int16)
        for ride_id in range(config.NUM_RIDES):
            idx = self._index_of[config.ride_node_id(ride_id)]
            self.node_idx_to_ride[idx] = ride_id

        self.ride_node_idx = np.array(
            [self._index_of[config.ride_node_id(r)] for r in range(config.NUM_RIDES)],
            dtype=np.int32,
        )

    @property
    def entrance_node(self) -> int:
        return config.NODE_ENTRANCE

    @property
    def node_ids(self) -> list[int]:
        return self._node_ids

    def node_to_idx(self, node_id: int) -> int:
        return self._index_of[node_id]

    def idx_to_node(self, node_idx: int) -> int:
        return int(self.idx_to_node_id[node_idx])

    def coords(self, node_id: int) -> tuple[float, float]:
        return self._graph.node_coords[node_id]

    def ride_node(self, ride_id: int) -> int:
        return config.ride_node_id(ride_id)

    def node_to_ride(self, node_id: int) -> int | None:
        return self._ride_node_to_id.get(node_id)

    def walk_time(self, from_node: int, to_node: int) -> int:
        i = self._index_of[from_node]
        j = self._index_of[to_node]
        return int(self.base_walk_matrix[i, j])

    def party_walk_sec(self, from_node_idx: int, to_node_idx: int, effective_speed: float) -> int:
        base = int(self.base_walk_matrix[from_node_idx, to_node_idx])
        return max(1, int(math.ceil(base * self._speed_scale / max(effective_speed, 0.1))))

    def party_walk_to_ride_sec(self, from_node_idx: int, ride_id: int, effective_speed: float) -> int:
        base = int(self.base_walk_to_rides[from_node_idx, ride_id])
        return max(1, int(math.ceil(base * self._speed_scale / max(effective_speed, 0.1))))

    def party_walk_time(self, from_node: int, to_node: int, effective_speed: float) -> int:
        return self.party_walk_sec(self._index_of[from_node], self._index_of[to_node], effective_speed)

    def walk_times_to_rides(self, from_node: int, effective_speed: float) -> np.ndarray:
        row_idx = self._index_of[from_node]
        scale = self._speed_scale / max(effective_speed, 0.1)
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

    def random_idle_node(self, rng, current_node: int, max_hops: int = config.IDLE_MAX_HOPS) -> int:
        candidates = self.neighbors_within_hops(current_node, max_hops)
        if not candidates:
            return current_node
        return int(rng.choice(candidates))

    def random_idle_node_idx(self, rng, current_node_idx: int, max_hops: int = config.IDLE_MAX_HOPS) -> int:
        node_id = self.idx_to_node(current_node_idx)
        idle = self.random_idle_node(rng, node_id, max_hops)
        return self.node_to_idx(idle)


_GRAPH: ParkGraph | None = None


def get_park_graph() -> ParkGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = ParkGraph()
    return _GRAPH
