#!/usr/bin/env python3
"""Post-process data/pathways.json: ride placements + light path cleanup.

- Rename/placement overrides for Space Mountain, Indiana, Small World, Autopia, Rise
- Buzz vertical spur to Tomorrowland spine
- Strip Main Street left/right parallels (keep center)
- Drop dead-end paths west of Rise
- Lightly thin the Small World plaza cluster

Usage (after a fresh extract):
    python tools/extract_osm_pathways.py   # calls this automatically
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx

PATH = ROOT / "data" / "pathways.json"

RISE = 0
INDIANA = 7
JUNGLE = 8
SMALL_WORLD = 24
SPACE = 28
BUZZ = 30
AUTOPIA = 32
MATTERHORN = 13

TL_CORRIDOR_Y_MIN = 560.0
TL_CORRIDOR_Y_MAX = 585.0
TL_CORRIDOR_X_MIN = 640.0
TL_CORRIDOR_X_MAX = 820.0

BUZZ_ONLY_LOOP_X_MIN = 750.0
BUZZ_ONLY_LOOP_X_MAX = 805.0
BUZZ_ONLY_LOOP_Y_MIN = 495.0
BUZZ_ONLY_LOOP_Y_MAX = 550.0

# Main Street band (entrance south → hub).
MS_Y_MIN = 600.0
MS_Y_MAX = 900.0
MS_MID_X_MIN = 562.0
MS_MID_X_MAX = 578.0


def _hypot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_node(nodes: dict, x: float, y: float) -> str:
    best, best_d = None, float("inf")
    for nid, nd in nodes.items():
        d = _hypot((nd["x"], nd["y"]), (x, y))
        if d < best_d:
            best_d = d
            best = nid
    assert best is not None
    return best


def _build_graph(data: dict) -> nx.Graph:
    g = nx.Graph()
    for nid, nd in data["nodes"].items():
        g.add_node(nid, **nd)
    for e in data["edges"]:
        g.add_edge(e["u"], e["v"], length_m=float(e["length_m"]), geometry=e["geometry"])
    return g


def _leaf_nodes(g: nx.Graph) -> list[str]:
    return [n for n in g.nodes() if g.degree(n) == 1]


def nudge_rides(g: nx.Graph, data: dict) -> None:
    rides = {int(r["ride_id"]): r for r in data["rides"]}

    # --- Space Mountain: tip of the stub almost directly above the old OSM spot ---
    sx = float(rides[SPACE]["x"])
    sy = float(rides[SPACE]["y"])
    above = []
    for n in _leaf_nodes(g):
        d = g.nodes[n]
        # Prefer tips close in x and not too far north of old Space.
        if sy - 120 <= d["y"] <= sy - 20 and abs(d["x"] - sx) < 25:
            above.append((abs(d["x"] - sx), -d["y"], n, d["x"], d["y"]))
    above.sort()  # closest x, then southernmost tip (largest y)
    if above:
        # Among tips within 12 display-units of Space's x, take southernmost.
        near = [t for t in above if t[0] < 12.0] or above
        near.sort(key=lambda t: t[1])  # most southern = smallest -y = largest y
        _, _, node, x, y = near[0]
        rides[SPACE]["x"] = round(x, 3)
        rides[SPACE]["y"] = round(y, 3)
        rides[SPACE]["snap_node"] = node
        rides[SPACE]["source"] = "simplified:above-stub"
        old = "1365116948"
        if old in g and old != node:
            cur = old
            seen = []
            while cur in g and g.degree(cur) <= 2 and cur not in {node}:
                nbrs = [n for n in g.neighbors(cur) if n not in seen]
                seen.append(cur)
                if len(nbrs) != 1:
                    break
                cur = nbrs[0]
            for n in seen:
                if n in g and n != node:
                    g.remove_node(n)

    # --- Indiana: southern tip of the Adventureland stub below Jungle Cruise ---
    jx = float(rides[JUNGLE]["x"])
    jy = float(rides[JUNGLE]["y"])
    south = []
    for n in _leaf_nodes(g):
        d = g.nodes[n]
        # South of Jungle, roughly in the Indy/Adventureland pocket.
        if d["y"] > jy + 15 and jx - 120 <= d["x"] <= jx + 20:
            south.append((d["y"], abs(d["x"] - (jx - 40)), n, d["x"], d["y"]))
    south.sort(reverse=True)
    if south:
        # Prefer the southernmost tip near x≈380–420.
        pool = [t for t in south if 360 <= t[3] <= 430] or south
        _, _, node, x, y = pool[0]
        rides[INDIANA]["x"] = round(x, 3)
        rides[INDIANA]["y"] = round(y, 3)
        rides[INDIANA]["snap_node"] = node
        rides[INDIANA]["source"] = "simplified:south-stub"

    # --- Rise: keep NW on GE network ---
    rides[RISE]["x"] = 130.0
    rides[RISE]["y"] = 250.0
    rides[RISE]["source"] = "simplified:northwest-ge"

    # --- Small World: deeper into the dense plaza cluster below the OSM marker ---
    swx, swy = float(rides[SMALL_WORLD]["x"]), float(rides[SMALL_WORLD]["y"])
    cluster = [
        (n, d["x"], d["y"])
        for n, d in g.nodes(data=True)
        if 690 <= d["x"] <= 800 and 230 <= d["y"] <= 290
    ]
    if cluster:
        best, best_c = None, -1
        for n, x, y in cluster:
            c = sum(1 for _, x2, y2 in cluster if _hypot((x, y), (x2, y2)) < 40)
            # Prefer slightly denser + more central-south points.
            score = c - 0.02 * abs(x - 720) - 0.01 * abs(y - 260)
            if score > best_c:
                best_c = score
                best = (n, x, y)
        if best:
            n, x, y = best
            rides[SMALL_WORLD]["x"] = round(x, 3)
            rides[SMALL_WORLD]["y"] = round(y, 3)
            rides[SMALL_WORLD]["snap_node"] = n
            rides[SMALL_WORLD]["source"] = "simplified:plaza-cluster"

    # --- Autopia: far-eastern Autopia track collection ---
    east = [
        (n, d["x"], d["y"])
        for n, d in g.nodes(data=True)
        if d["x"] > 920 and 470 <= d["y"] <= 550
    ]
    if east:
        best, best_c = None, -1
        for n, x, y in east:
            c = sum(1 for _, x2, y2 in east if _hypot((x, y), (x2, y2)) < 45)
            if c > best_c:
                best_c = c
                best = (n, x, y)
        if best:
            n, x, y = best
            rides[AUTOPIA]["x"] = round(x, 3)
            rides[AUTOPIA]["y"] = round(y, 3)
            rides[AUTOPIA]["snap_node"] = n
            rides[AUTOPIA]["source"] = "simplified:east-cluster"

    # --- Buzz: north of TL spine ---
    buzz = rides[BUZZ]
    corridor_y = 0.5 * (TL_CORRIDOR_Y_MIN + TL_CORRIDOR_Y_MAX)
    buzz["x"] = round(float(buzz["x"]), 3)
    buzz["y"] = round(corridor_y - 48.0, 3)
    buzz["source"] = "simplified:vertical-spur"


def fix_buzz_vertical_spur(g: nx.Graph, data: dict) -> None:
    rides = {int(r["ride_id"]): r for r in data["rides"]}
    buzz = rides[BUZZ]
    bx, by = float(buzz["x"]), float(buzz["y"])
    scale = float(data["meta"].get("meters_to_display_scale") or 1.3155)

    corridor = {
        n
        for n, d in g.nodes(data=True)
        if TL_CORRIDOR_X_MIN <= d["x"] <= TL_CORRIDOR_X_MAX
        and TL_CORRIDOR_Y_MIN <= d["y"] <= TL_CORRIDOR_Y_MAX
    }
    best, best_dx = None, float("inf")
    for n in corridor:
        dx = abs(g.nodes[n]["x"] - bx)
        if dx < best_dx:
            best_dx = dx
            best = n

    foot_id, buzz_id = "synthetic:buzz_foot", "synthetic:buzz"
    foot_y = g.nodes[best]["y"] if best is not None else 0.5 * (TL_CORRIDOR_Y_MIN + TL_CORRIDOR_Y_MAX)
    foot_x = bx

    g.add_node(buzz_id, x=bx, y=by, lat=None, lon=None)
    g.add_node(foot_id, x=round(foot_x, 3), y=round(foot_y, 3), lat=None, lon=None)
    g.add_edge(
        buzz_id,
        foot_id,
        length_m=round(abs(foot_y - by) / scale, 3),
        geometry=[[round(bx, 2), round(by, 2)], [round(foot_x, 2), round(foot_y, 2)]],
    )
    if best is None:
        best = _nearest_node({n: g.nodes[n] for n in g.nodes() if n not in {buzz_id, foot_id}}, foot_x, foot_y)

    g.add_edge(
        foot_id,
        best,
        length_m=round(max(abs(g.nodes[best]["x"] - foot_x) / scale, 0.5), 3),
        geometry=[
            [round(foot_x, 2), round(foot_y, 2)],
            [round(g.nodes[best]["x"], 2), round(g.nodes[best]["y"], 2)],
        ],
    )
    buzz["snap_node"] = buzz_id

    protect: set[str] = {buzz_id, foot_id, best}
    protect_snaps = []
    for rid in (MATTERHORN, 29, 31, AUTOPIA, 33, 34, SPACE):
        for r in data["rides"]:
            if int(r["ride_id"]) == rid and r["snap_node"] in g:
                protect_snaps.append(r["snap_node"])
                protect.add(r["snap_node"])
    for i, a in enumerate(protect_snaps):
        for b in protect_snaps[i + 1 :]:
            try:
                protect.update(nx.shortest_path(g, a, b, weight="length_m"))
            except nx.NetworkXNoPath:
                pass

    for n, d in list(g.nodes(data=True)):
        if str(n).startswith("synthetic:") or n in protect:
            continue
        if (
            BUZZ_ONLY_LOOP_X_MIN <= d["x"] <= BUZZ_ONLY_LOOP_X_MAX
            and BUZZ_ONLY_LOOP_Y_MIN <= d["y"] <= BUZZ_ONLY_LOOP_Y_MAX
        ):
            g.remove_node(n)


def strip_main_street_side_lanes(g: nx.Graph, data: dict) -> int:
    """Remove left/right Main Street parallels; keep the center vertical."""
    keep = {str(r["snap_node"]) for r in data["rides"]}
    keep |= {str(h["snap_node"]) for h in data["hubs"].values()}
    keep |= {n for n in g.nodes() if str(n).startswith("synthetic:")}
    # Always protect entrance + main hub.
    for key in ("entrance", "main_hub", "central_plaza"):
        if key in data["hubs"]:
            keep.add(str(data["hubs"][key]["snap_node"]))

    removed = 0
    for n, d in list(g.nodes(data=True)):
        if n in keep or str(n).startswith("synthetic:"):
            continue
        x, y = d["x"], d["y"]
        if not (MS_Y_MIN <= y <= MS_Y_MAX):
            continue
        if 480 <= x < MS_MID_X_MIN or MS_MID_X_MAX < x <= 640:
            g.remove_node(n)
            removed += 1
    return removed


def prune_dead_ends_west_of_rise(g: nx.Graph, data: dict) -> int:
    """Delete nowhere paths to the left of Rise."""
    rides = {int(r["ride_id"]): r for r in data["rides"]}
    rx = float(rides[RISE]["x"])
    ry = float(rides[RISE]["y"])
    keep = {str(r["snap_node"]) for r in data["rides"]}
    keep |= {str(h["snap_node"]) for h in data["hubs"].values()}
    keep |= {n for n in g.nodes() if str(n).startswith("synthetic:")}

    removed = 0
    # Hard-delete nodes in a box immediately west of Rise (the nowhere scribble).
    for n, d in list(g.nodes(data=True)):
        if n in keep:
            continue
        if d["x"] < rx - 2 and ry - 40 <= d["y"] <= ry + 90:
            g.remove_node(n)
            removed += 1

    candidates = [
        n
        for n, d in g.nodes(data=True)
        if d["x"] < rx - 8 and 200 <= d["y"] <= 480 and n not in keep
    ]
    sub = g.subgraph(candidates).copy()
    for comp in list(nx.connected_components(sub)):
        g.remove_nodes_from(comp)
        removed += len(comp)

    changed = True
    while changed:
        changed = False
        for n in list(g.nodes()):
            if n in keep:
                continue
            d = g.nodes[n]
            if d["x"] >= rx - 5:
                continue
            if g.degree(n) <= 1:
                g.remove_node(n)
                removed += 1
                changed = True
    return removed


def lightly_simplify_small_world_cluster(g: nx.Graph, data: dict) -> int:
    """Slightly thin the dense plaza where Small World now sits."""
    rides = {int(r["ride_id"]): r for r in data["rides"]}
    sw = rides[SMALL_WORLD]
    cx, cy = float(sw["x"]), float(sw["y"])
    keep = {str(r["snap_node"]) for r in data["rides"]}
    keep |= {str(h["snap_node"]) for h in data["hubs"].values()}
    keep |= {n for n in g.nodes() if str(n).startswith("synthetic:")}

    cluster = [
        n
        for n, d in g.nodes(data=True)
        if _hypot((d["x"], d["y"]), (cx, cy)) < 70
    ]
    removed = 0
    # Remove degree-2 nodes that are nearly colinear shortcuts (every other one).
    deg2 = [n for n in cluster if n not in keep and g.degree(n) == 2]
    # Sort by angle around centroid and drop every 3rd.
    def ang(n: str) -> float:
        d = g.nodes[n]
        return math.atan2(d["y"] - cy, d["x"] - cx)

    deg2.sort(key=ang)
    for i, n in enumerate(deg2):
        if i % 2 != 0:
            continue
        if n not in g or g.degree(n) != 2:
            continue
        a, b = list(g.neighbors(n))
        # Bridge a-b if missing, then drop n.
        if not g.has_edge(a, b):
            length = g[n][a].get("length_m", 1.0) + g[n][b].get("length_m", 1.0)
            g.add_edge(
                a,
                b,
                length_m=length,
                geometry=[
                    [round(g.nodes[a]["x"], 2), round(g.nodes[a]["y"], 2)],
                    [round(g.nodes[b]["x"], 2), round(g.nodes[b]["y"], 2)],
                ],
            )
        g.remove_node(n)
        removed += 1
    return removed


def resnap_rides_and_hubs(g: nx.Graph, data: dict) -> None:
    coords = {n: g.nodes[n] for n in g.nodes()}
    for ride in data["rides"]:
        snap = ride.get("snap_node")
        if str(snap).startswith("synthetic:") and snap in g:
            continue
        # Keep explicit stub snaps when the node still exists.
        if (
            ride.get("source", "").startswith("simplified:")
            and snap in g
            and not str(snap).startswith("simplified")
        ):
            continue
        ride["snap_node"] = _nearest_node(coords, float(ride["x"]), float(ride["y"]))
    for hub in data["hubs"].values():
        hub["snap_node"] = _nearest_node(coords, float(hub["x"]), float(hub["y"]))


def graph_to_payload(g: nx.Graph, data: dict) -> dict:
    nodes = {
        str(n): {
            "x": round(float(d["x"]), 3),
            "y": round(float(d["y"]), 3),
            "lat": d.get("lat"),
            "lon": d.get("lon"),
        }
        for n, d in g.nodes(data=True)
    }
    edges = []
    seen: set[tuple[str, str]] = set()
    for u, v, d in g.edges(data=True):
        su, sv = str(u), str(v)
        key = (su, sv) if su < sv else (sv, su)
        if key in seen:
            continue
        seen.add(key)
        geom = d.get("geometry") or [
            [round(g.nodes[u]["x"], 2), round(g.nodes[u]["y"], 2)],
            [round(g.nodes[v]["x"], 2), round(g.nodes[v]["y"], 2)],
        ]
        edges.append(
            {"u": su, "v": sv, "length_m": round(float(d.get("length_m", 1.0)), 3), "geometry": geom}
        )
    out = dict(data)
    out["nodes"] = nodes
    out["edges"] = edges
    out["meta"] = dict(data["meta"])
    out["meta"]["num_nodes"] = len(nodes)
    out["meta"]["num_edges"] = len(edges)
    out["meta"]["simplified"] = True
    out["meta"]["simplify_level"] = "layout-v3"
    return out


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    if data.get("meta", {}).get("simplified"):
        raise SystemExit(
            f"{PATH} is already simplified. Re-run tools/extract_osm_pathways.py first."
        )

    # Keep ride names in pathways.json in sync with config rename.
    for r in data["rides"]:
        if r.get("name") == "Hyperspace Mountain":
            r["name"] = "Space Mountain"

    before_n, before_e = len(data["nodes"]), len(data["edges"])
    g = _build_graph(data)
    nudge_rides(g, data)
    fix_buzz_vertical_spur(g, data)

    ms_removed = strip_main_street_side_lanes(g, data)
    west_removed = prune_dead_ends_west_of_rise(g, data)
    sw_removed = lightly_simplify_small_world_cluster(g, data)

    resnap_rides_and_hubs(g, data)

    # Ensure explicit snaps still exist; otherwise nearest.
    coords = {n: g.nodes[n] for n in g.nodes()}
    for ride in data["rides"]:
        if ride["snap_node"] not in g:
            ride["snap_node"] = _nearest_node(coords, float(ride["x"]), float(ride["y"]))

    if nx.number_connected_components(g) != 1:
        entrance = data["hubs"]["entrance"]["snap_node"]
        keep = {str(r["snap_node"]) for r in data["rides"]}
        keep |= {str(h["snap_node"]) for h in data["hubs"].values()}
        keep |= {n for n in g.nodes() if str(n).startswith("synthetic:")}
        comps = [set(c) for c in nx.connected_components(g)]
        main = next((c for c in comps if entrance in c), max(comps, key=len))
        scale = float(data["meta"].get("meters_to_display_scale") or 1.3155)
        for c in comps:
            if c == main:
                continue
            if not (c & keep):
                g.remove_nodes_from(c)
                continue
            best, best_d = None, float("inf")
            for a in c:
                for b in main:
                    d = _hypot((g.nodes[a]["x"], g.nodes[a]["y"]), (g.nodes[b]["x"], g.nodes[b]["y"]))
                    if d < best_d:
                        best_d, best = d, (a, b)
            if best:
                a, b = best
                g.add_edge(
                    a,
                    b,
                    length_m=round(max(best_d / scale, 0.5), 3),
                    geometry=[
                        [round(g.nodes[a]["x"], 2), round(g.nodes[a]["y"], 2)],
                        [round(g.nodes[b]["x"], 2), round(g.nodes[b]["y"], 2)],
                    ],
                )
                main |= c

    out = graph_to_payload(g, data)
    PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"Simplified {PATH}: nodes {before_n}->{out['meta']['num_nodes']}, "
        f"edges {before_e}->{out['meta']['num_edges']} "
        f"(ms={ms_removed}, west={west_removed}, sw={sw_removed})"
    )
    for rid in (RISE, INDIANA, SMALL_WORLD, SPACE, BUZZ, AUTOPIA):
        r = next(x for x in out["rides"] if int(x["ride_id"]) == rid)
        print(f"  ride {rid} {r['name']}: ({r['x']}, {r['y']}) snap={r['snap_node']}")


if __name__ == "__main__":
    main()
