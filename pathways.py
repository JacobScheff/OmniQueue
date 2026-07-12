"""OSM-derived walkway network for path lengths and curved layout geometry.

Loads committed ``data/pathways.json`` (produced by ``tools/extract_osm_pathways.py``).
Routing nodes (hubs / rides) snap to nearest pathway nodes; walk distances follow
the pedestrian network in meters rather than straight-line hub edges.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx

PATHWAYS_PATH = Path(__file__).resolve().parent / "data" / "pathways.json"
RIDE_NODE_OFFSET = 100  # must match config.RIDE_NODE_OFFSET


class PathwayNetwork:
    """Undirected walkway graph with display polylines and meter lengths."""

    def __init__(self, data: dict) -> None:
        self.meta = data["meta"]
        self.nodes: dict[str, dict] = data["nodes"]
        self.edges_raw: list[dict] = data["edges"]
        self.rides: list[dict] = data["rides"]
        self.hubs: dict[str, dict] = data["hubs"]

        self.graph = nx.Graph()
        for nid, nd in self.nodes.items():
            self.graph.add_node(nid, x=nd["x"], y=nd["y"])

        self.edge_geometry: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for e in self.edges_raw:
            u, v = e["u"], e["v"]
            length = float(e["length_m"])
            geom = [(float(x), float(y)) for x, y in e["geometry"]]
            self.graph.add_edge(u, v, length_m=length)
            self.edge_geometry[(u, v)] = geom
            self.edge_geometry[(v, u)] = list(reversed(geom))

        self.ride_snap: list[str] = [r["snap_node"] for r in self.rides]
        self.hub_snap: dict[int, str] = {
            int(h["node_id"]): h["snap_node"] for h in self.hubs.values()
        }
        self.ride_coords: list[tuple[float, float]] = [
            (float(r["x"]), float(r["y"])) for r in self.rides
        ]
        self.hub_coords: dict[int, tuple[float, float]] = {
            int(h["node_id"]): (float(h["x"]), float(h["y"])) for h in self.hubs.values()
        }

        # Short spur from POI display position → snapped pathway node.
        self._spur_m: dict[int, float] = {}
        for rid, ride in enumerate(self.rides):
            snap = self.nodes[ride["snap_node"]]
            self._spur_m[RIDE_NODE_OFFSET + rid] = _display_dist_m(
                (ride["x"], ride["y"]),
                (snap["x"], snap["y"]),
                self.meta,
            )
        for hub in self.hubs.values():
            nid = int(hub["node_id"])
            snap = self.nodes[hub["snap_node"]]
            self._spur_m[nid] = _display_dist_m(
                (hub["x"], hub["y"]),
                (snap["x"], snap["y"]),
                self.meta,
            )

    def snap_for_routing_node(self, node_id: int) -> str:
        if node_id >= RIDE_NODE_OFFSET:
            return self.ride_snap[node_id - RIDE_NODE_OFFSET]
        return self.hub_snap[node_id]

    def path_length_m(self, from_node: int, to_node: int) -> float:
        if from_node == to_node:
            return 0.0
        u = self.snap_for_routing_node(from_node)
        v = self.snap_for_routing_node(to_node)
        try:
            network_m = float(nx.shortest_path_length(self.graph, u, v, weight="length_m"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            ca = self.coords_for_routing_node(from_node)
            cb = self.coords_for_routing_node(to_node)
            network_m = _display_dist_m(ca, cb, self.meta)
        return network_m + self._spur_m.get(from_node, 0.0) + self._spur_m.get(to_node, 0.0)

    def path_polyline(self, from_node: int, to_node: int) -> list[tuple[float, float]]:
        """Display-coordinate polyline along walkways from routing node to routing node."""
        start = self.coords_for_routing_node(from_node)
        end = self.coords_for_routing_node(to_node)
        if from_node == to_node:
            return [start]

        u = self.snap_for_routing_node(from_node)
        v = self.snap_for_routing_node(to_node)
        try:
            node_path = nx.shortest_path(self.graph, u, v, weight="length_m")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [start, end]

        poly: list[tuple[float, float]] = [start]
        snap_u = (self.nodes[u]["x"], self.nodes[u]["y"])
        if _hypot(start, snap_u) > 0.5:
            poly.append(snap_u)

        for a, b in zip(node_path, node_path[1:]):
            geom = self.edge_geometry.get((a, b))
            if geom and len(geom) >= 2:
                pts = (
                    geom
                    if not poly or _hypot(poly[-1], geom[0]) <= _hypot(poly[-1], geom[-1])
                    else list(reversed(geom))
                )
                for pt in pts[1:]:
                    poly.append(pt)
            else:
                nb = (self.nodes[b]["x"], self.nodes[b]["y"])
                poly.append(nb)

        snap_v = (self.nodes[v]["x"], self.nodes[v]["y"])
        if _hypot(end, snap_v) > 0.5:
            poly.append(end)
        elif _hypot(poly[-1], end) > 0.5:
            poly.append(end)
        return poly

    def coords_for_routing_node(self, node_id: int) -> tuple[float, float]:
        if node_id >= RIDE_NODE_OFFSET:
            return self.ride_coords[node_id - RIDE_NODE_OFFSET]
        return self.hub_coords[node_id]

    def all_edge_polylines(self) -> list[list[tuple[float, float]]]:
        """Unique undirected edge geometries for map drawing."""
        seen: set[tuple[str, str]] = set()
        out: list[list[tuple[float, float]]] = []
        for e in self.edges_raw:
            key = (e["u"], e["v"]) if e["u"] < e["v"] else (e["v"], e["u"])
            if key in seen:
                continue
            seen.add(key)
            out.append([(float(x), float(y)) for x, y in e["geometry"]])
        return out


def _hypot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _display_dist_m(
    a: tuple[float, float], b: tuple[float, float], meta: dict
) -> float:
    scale = float(meta.get("meters_to_display_scale") or 1.0)
    return _hypot(a, b) / max(scale, 1e-9)


@lru_cache(maxsize=1)
def load_pathways() -> PathwayNetwork | None:
    if not PATHWAYS_PATH.is_file():
        return None
    data = json.loads(PATHWAYS_PATH.read_text(encoding="utf-8"))
    return PathwayNetwork(data)


def apply_pathway_coords(cfg: Any) -> bool:
    """Overwrite hub/ride display coords on a config module from pathways data."""
    net = load_pathways()
    if net is None:
        return False
    for rid, (x, y) in enumerate(net.ride_coords):
        cfg.RIDES[rid]["coords"] = (x, y)
    for nid, xy in net.hub_coords.items():
        cfg.HUB_COORDS[nid] = xy
    if cfg.NODE_ENTRANCE in net.hub_coords:
        cfg.ENTRANCE_COORDS = net.hub_coords[cfg.NODE_ENTRANCE]
    return True


def interpolate_polyline(
    points: list[tuple[float, float]], progress: float
) -> tuple[float, float]:
    """Interpolate ``progress`` in [0, 1] along a polyline by arc length."""
    if not points:
        return (0.0, 0.0)
    if len(points) == 1 or progress <= 0:
        return points[0]
    if progress >= 1:
        return points[-1]

    segs = [_hypot(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total = sum(segs)
    if total <= 1e-9:
        return points[0]
    target = progress * total
    acc = 0.0
    for i, length in enumerate(segs):
        if acc + length >= target:
            t = 0.0 if length <= 1e-9 else (target - acc) / length
            ax, ay = points[i]
            bx, by = points[i + 1]
            return ax + (bx - ax) * t, ay + (by - ay) * t
        acc += length
    return points[-1]
