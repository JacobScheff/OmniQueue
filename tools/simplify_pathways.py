#!/usr/bin/env python3
"""Minimal post-process for data/pathways.json.

Preserves nearly all OSM walkway detail. Only:
  - nudges Indiana next to Jungle Cruise
  - places Rise in northwest Galaxy's Edge
  - replaces Buzz's plaza loop approach with a vertical spur
  - lightly strips the eastern Buzz-only plaza bubble (not Matterhorn corridors)
  - prunes true dangling leaves that are not ride/hub snaps

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
BUZZ = 30
MATTERHORN = 13

TL_CORRIDOR_Y_MIN = 560.0
TL_CORRIDOR_Y_MAX = 585.0
TL_CORRIDOR_X_MIN = 640.0
TL_CORRIDOR_X_MAX = 820.0

# Eastern Autopia bubble only — do NOT include Matterhorn approaches (~x=650–720).
BUZZ_ONLY_LOOP_X_MIN = 750.0
BUZZ_ONLY_LOOP_X_MAX = 805.0
BUZZ_ONLY_LOOP_Y_MIN = 495.0
BUZZ_ONLY_LOOP_Y_MAX = 550.0


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


def prune_dangling(g: nx.Graph, keep: set[str]) -> int:
    removed = 0
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes()):
            if n in keep:
                continue
            if g.degree(n) <= 1:
                g.remove_node(n)
                removed += 1
                changed = True
    return removed


def nudge_rides(data: dict) -> None:
    rides = {int(r["ride_id"]): r for r in data["rides"]}

    # Indiana right next to Jungle Cruise (slightly SW).
    jx, jy = float(rides[JUNGLE]["x"]), float(rides[JUNGLE]["y"])
    rides[INDIANA]["x"] = round(jx - 12.0, 3)
    rides[INDIANA]["y"] = round(jy + 6.0, 3)
    rides[INDIANA]["source"] = "simplified:near-jungle-cruise"

    # Rise: northwest on the GE walk network (OSM has little coverage north of ~y=245).
    # Sit on the NW corridor so Critter↔GE paths meet the ride marker.
    rides[RISE]["x"] = 130.0
    rides[RISE]["y"] = 250.0
    rides[RISE]["source"] = "simplified:northwest-ge"

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

    # Protect Matterhorn ↔ Tomorrowland paths; only remove eastern Buzz plaza nodes
    # that are not on those corridors.
    protect: set[str] = {buzz_id, foot_id, best}
    protect_snaps = []
    for rid in (MATTERHORN, 29, 31, 32, 33, 34):
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
        if str(n).startswith("synthetic:"):
            continue
        if n in protect:
            continue
        if (
            BUZZ_ONLY_LOOP_X_MIN <= d["x"] <= BUZZ_ONLY_LOOP_X_MAX
            and BUZZ_ONLY_LOOP_Y_MIN <= d["y"] <= BUZZ_ONLY_LOOP_Y_MAX
        ):
            g.remove_node(n)


def resnap_rides_and_hubs(g: nx.Graph, data: dict) -> None:
    coords = {n: g.nodes[n] for n in g.nodes()}
    for ride in data["rides"]:
        if str(ride.get("snap_node", "")).startswith("synthetic:") and ride["snap_node"] in g:
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
    out["meta"]["simplify_level"] = "minimal"
    return out


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    if data.get("meta", {}).get("simplified"):
        raise SystemExit(
            f"{PATH} is already simplified. Re-run tools/extract_osm_pathways.py first."
        )

    before_n, before_e = len(data["nodes"]), len(data["edges"])
    nudge_rides(data)
    g = _build_graph(data)
    fix_buzz_vertical_spur(g, data)

    keep = {str(r["snap_node"]) for r in data["rides"]}
    keep |= {str(h["snap_node"]) for h in data["hubs"].values()}
    keep |= {n for n in g.nodes() if str(n).startswith("synthetic:")}

    # Do not mass-prune dangling leaves — that erased Critter↔GE spurs and
    # secondary walkways the map needs. Only drop degree-0 isolates.
    removed = 0
    for n in list(g.nodes()):
        if n not in keep and g.degree(n) == 0:
            g.remove_node(n)
            removed += 1

    resnap_rides_and_hubs(g, data)
    keep = {str(r["snap_node"]) for r in data["rides"]}
    keep |= {str(h["snap_node"]) for h in data["hubs"].values()}
    keep |= {n for n in g.nodes() if str(n).startswith("synthetic:")}

    if nx.number_connected_components(g) != 1:
        # Bridge orphans that still hold ride snaps.
        entrance = data["hubs"]["entrance"]["snap_node"]
        comps = [set(c) for c in nx.connected_components(g)]
        main = next(c for c in comps if entrance in c)
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
        f"edges {before_e}->{out['meta']['num_edges']} (dangling={removed})"
    )
    for rid in (RISE, INDIANA, JUNGLE, BUZZ, 2, 3, MATTERHORN):
        r = next(x for x in out["rides"] if int(x["ride_id"]) == rid)
        print(f"  ride {rid} {r['name']}: ({r['x']}, {r['y']}) snap={r['snap_node']}")


if __name__ == "__main__":
    main()
