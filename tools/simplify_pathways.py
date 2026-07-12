#!/usr/bin/env python3
"""Slightly simplify data/pathways.json: prune dangling branches, nudge rides.

Safe to re-run only on a fresh extract (meta.simplified must be absent/false).
Prefer: python tools/extract_osm_pathways.py && python tools/simplify_pathways.py

Usage:
    python tools/simplify_pathways.py
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
HAUNTED = 5
PIRATES = 6
INDIANA = 7
BUZZ = 30

RISE_INWARD_UNITS = 120.0
INDIANA_PIRATES_WEIGHT = 0.55
# Pad around ride/hub bbox when dropping exterior fringe nodes.
BBOX_PAD = 22.0
TL_CORRIDOR_Y_MIN = 560.0
TL_CORRIDOR_Y_MAX = 585.0
TL_CORRIDOR_X_MIN = 640.0
TL_CORRIDOR_X_MAX = 820.0
# Strip the Observatron/Autopia plaza loop north of the TL spine (Buzz approach).
BUZZ_LOOP_X_MIN = 670.0
BUZZ_LOOP_X_MAX = 810.0
BUZZ_LOOP_Y_MIN = 460.0
BUZZ_LOOP_Y_MAX = 558.0


def _hypot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_node(nodes: dict, x: float, y: float, allowed: set[str] | None = None) -> str:
    best, best_d = None, float("inf")
    for nid, nd in nodes.items():
        if allowed is not None and nid not in allowed:
            continue
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


def _keep_nodes(data: dict) -> set[str]:
    keep = {r["snap_node"] for r in data["rides"]}
    keep |= {h["snap_node"] for h in data["hubs"].values()}
    keep |= {n for n in keep if True}
    return {str(n) for n in keep}


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


def prune_outside_bbox(g: nx.Graph, data: dict, keep: set[str], pad: float) -> int:
    xs = [float(r["x"]) for r in data["rides"]] + [float(h["x"]) for h in data["hubs"].values()]
    ys = [float(r["y"]) for r in data["rides"]] + [float(h["y"]) for h in data["hubs"].values()]
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    removed = 0
    for n in list(g.nodes()):
        if n in keep:
            continue
        nd = g.nodes[n]
        if not (min_x <= nd["x"] <= max_x and min_y <= nd["y"] <= max_y):
            g.remove_node(n)
            removed += 1
    return removed


def prune_useless_not_on_keep_paths(g: nx.Graph, keep: set[str], sample_limit: int = 80) -> int:
    """Remove nodes that never appear on shortest paths between keep nodes.

    Only a light pass: if the useful set is too small (<50% of graph), skip.
    """
    present = [n for n in keep if n in g]
    if len(present) < 2:
        return 0

    useful: set[str] = set(present)
    # Limit pair count for speed; still covers park connectivity.
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            try:
                useful.update(nx.shortest_path(g, a, b, weight="length_m"))
            except nx.NetworkXNoPath:
                pass
        if i >= sample_limit:
            break

    if len(useful) < 0.45 * g.number_of_nodes():
        # Too aggressive / disconnected — do not apply.
        return 0

    removed = 0
    for n in list(g.nodes()):
        if n in useful or n in keep:
            continue
        g.remove_node(n)
        removed += 1
    return removed


def nudge_rides(data: dict) -> None:
    rides = {int(r["ride_id"]): r for r in data["rides"]}

    pirates = rides[PIRATES]
    haunted = rides[HAUNTED]
    jungle = rides[8]  # Jungle Cruise — park-inward reference
    w = INDIANA_PIRATES_WEIGHT
    # Between Pirates and Haunted, closer to Pirates, but pulled east toward Jungle Cruise.
    rides[INDIANA]["x"] = round(
        0.50 * pirates["x"] + 0.20 * haunted["x"] + 0.30 * jungle["x"], 3
    )
    rides[INDIANA]["y"] = round(
        0.55 * pirates["y"] + 0.25 * haunted["y"] + 0.20 * jungle["y"], 3
    )
    rides[INDIANA]["source"] = "simplified:between-pirates-haunted"

    rise = rides[RISE]
    # Place Rise on the inner Galaxy's Edge approach (east of the outer loop).
    rise["x"] = round(max(float(rise["x"]), 240.0), 3)
    rise["y"] = round(min(max(float(rise["y"]), 250.0), 320.0) + 40.0, 3)
    # Then pull toward Critter / park interior.
    target = (300.0, 360.0)
    dx, dy = target[0] - rise["x"], target[1] - rise["y"]
    norm = math.hypot(dx, dy) or 1.0
    rise["x"] = round(rise["x"] + dx / norm * 55.0, 3)
    rise["y"] = round(rise["y"] + dy / norm * 55.0, 3)
    rise["source"] = "simplified:inward"

    buzz = rides[BUZZ]
    corridor_y = 0.5 * (TL_CORRIDOR_Y_MIN + TL_CORRIDOR_Y_MAX)
    buzz["x"] = round(float(buzz["x"]), 3)
    buzz["y"] = round(corridor_y - 48.0, 3)
    buzz["source"] = "simplified:vertical-spur"


def prune_outward_spurs(g: nx.Graph, keep: set[str], center: tuple[float, float]) -> int:
    """Remove short outward whiskers: degree-1/2 chains pointing away from park center."""
    removed = 0
    changed = True
    cx, cy = center
    while changed:
        changed = False
        for n in list(g.nodes()):
            if n in keep:
                continue
            deg = g.degree(n)
            if deg == 0:
                g.remove_node(n)
                removed += 1
                changed = True
                continue
            if deg > 2:
                continue
            x, y = g.nodes[n]["x"], g.nodes[n]["y"]
            dist_n = math.hypot(x - cx, y - cy)
            # If all neighbors are closer to center, this node is an outward tip.
            neighbors = list(g.neighbors(n))
            if not neighbors:
                continue
            if all(
                math.hypot(g.nodes[nb]["x"] - cx, g.nodes[nb]["y"] - cy) < dist_n - 1.0
                for nb in neighbors
            ):
                g.remove_node(n)
                removed += 1
                changed = True
    return removed


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

    best = None
    best_dx = float("inf")
    for n in corridor:
        dx = abs(g.nodes[n]["x"] - bx)
        if dx < best_dx:
            best_dx = dx
            best = n

    foot_id = "synthetic:buzz_foot"
    buzz_id = "synthetic:buzz"
    if best is not None and best_dx < 25:
        foot_x = 0.25 * g.nodes[best]["x"] + 0.75 * bx
        foot_y = g.nodes[best]["y"]
        attach = best
    else:
        foot_x = bx
        foot_y = 0.5 * (TL_CORRIDOR_Y_MIN + TL_CORRIDOR_Y_MAX)
        attach = None

    old_snap = buzz.get("snap_node")

    g.add_node(buzz_id, x=bx, y=by, lat=None, lon=None)
    g.add_node(foot_id, x=round(foot_x, 3), y=round(foot_y, 3), lat=None, lon=None)

    spur_len_m = abs(foot_y - by) / scale
    g.add_edge(
        buzz_id,
        foot_id,
        length_m=round(spur_len_m, 3),
        geometry=[[round(bx, 2), round(by, 2)], [round(foot_x, 2), round(foot_y, 2)]],
    )

    if attach is None:
        attach = _nearest_node(
            {n: g.nodes[n] for n in g.nodes() if n not in {buzz_id, foot_id}},
            foot_x,
            foot_y,
        )

    attach_len = _hypot((foot_x, foot_y), (g.nodes[attach]["x"], g.nodes[attach]["y"]))
    g.add_edge(
        foot_id,
        attach,
        length_m=round(max(attach_len / scale, 0.5), 3),
        geometry=[
            [round(foot_x, 2), round(foot_y, 2)],
            [round(g.nodes[attach]["x"], 2), round(g.nodes[attach]["y"], 2)],
        ],
    )
    buzz["snap_node"] = buzz_id

    # Wipe the plaza/Observatron loop north of the TL spine so Buzz is only a
    # vertical spur. Autopia / Monorail sit east of the box; Nemo is resnapped.
    loop_box = [
        n
        for n, d in g.nodes(data=True)
        if BUZZ_LOOP_X_MIN <= d["x"] <= BUZZ_LOOP_X_MAX
        and BUZZ_LOOP_Y_MIN <= d["y"] <= BUZZ_LOOP_Y_MAX
        and not str(n).startswith("synthetic:")
    ]
    g.remove_nodes_from(loop_box)

    # Collapse the dual east-west Tomorrowland tracks (y≈571 and y≈583) into the
    # northern spine so Buzz's foot does not land in a rectangular junction.
    north_band = {
        n
        for n, d in g.nodes(data=True)
        if TL_CORRIDOR_X_MIN <= d["x"] <= TL_CORRIDOR_X_MAX
        and 568.0 <= d["y"] <= 576.0
    }
    south_band = {
        n
        for n, d in g.nodes(data=True)
        if TL_CORRIDOR_X_MIN <= d["x"] <= TL_CORRIDOR_X_MAX
        and 578.0 <= d["y"] <= 600.0
        and not str(n).startswith("synthetic:")
    }
    # Remap south-band snaps (e.g. Star Tours) onto nearest north-band node, then drop south band.
    if north_band and south_band:
        for ride in data["rides"]:
            if ride.get("snap_node") in south_band:
                sx, sy = float(ride["x"]), float(ride["y"])
                ride["snap_node"] = _nearest_node(
                    {n: g.nodes[n] for n in north_band}, sx, sy
                )
        for hub in data["hubs"].values():
            if hub.get("snap_node") in south_band:
                hub["snap_node"] = _nearest_node(
                    {n: g.nodes[n] for n in north_band},
                    float(hub["x"]),
                    float(hub["y"]),
                )
        # Drop south-band nodes not required as the only bridge elsewhere.
        g.remove_nodes_from(south_band)

    # Re-pick Buzz foot on the remaining single corridor.
    corridor = {
        n
        for n, d in g.nodes(data=True)
        if TL_CORRIDOR_X_MIN <= d["x"] <= TL_CORRIDOR_X_MAX
        and TL_CORRIDOR_Y_MIN <= d["y"] <= 576.0
    }
    best = None
    best_dx = float("inf")
    for n in corridor:
        dx = abs(g.nodes[n]["x"] - bx)
        if dx < best_dx:
            best_dx = dx
            best = n
    if best is not None:
        # Pure vertical: attach Buzz directly to a point on the corridor at Buzz's x.
        foot_x = bx
        foot_y = g.nodes[best]["y"]
        g.nodes[foot_id]["x"] = round(foot_x, 3)
        g.nodes[foot_id]["y"] = round(foot_y, 3)
        for nbr in list(g.neighbors(foot_id)):
            if nbr != buzz_id:
                g.remove_edge(foot_id, nbr)
        if g.has_edge(buzz_id, foot_id):
            g[buzz_id][foot_id]["geometry"] = [
                [round(bx, 2), round(by, 2)],
                [round(foot_x, 2), round(foot_y, 2)],
            ]
            g[buzz_id][foot_id]["length_m"] = round(abs(foot_y - by) / scale, 3)
        # Splice foot into the corridor between best and its horizontal neighbors.
        if best != foot_id and best in g:
            nbrs = [n for n in g.neighbors(best) if n != foot_id]
            # Connect foot to best when they differ in x; otherwise reconnect best's nbrs to foot.
            if abs(g.nodes[best]["x"] - foot_x) < 1.0:
                for nbr in nbrs:
                    if nbr == buzz_id:
                        continue
                    length = _hypot((foot_x, foot_y), (g.nodes[nbr]["x"], g.nodes[nbr]["y"]))
                    g.add_edge(
                        foot_id,
                        nbr,
                        length_m=round(max(length / scale, 0.5), 3),
                        geometry=[
                            [round(foot_x, 2), round(foot_y, 2)],
                            [round(g.nodes[nbr]["x"], 2), round(g.nodes[nbr]["y"], 2)],
                        ],
                    )
                # Drop best if it became redundant and isn't a keep snap.
                keep_snaps = {r["snap_node"] for r in data["rides"]} | {
                    h["snap_node"] for h in data["hubs"].values()
                }
                if best not in keep_snaps and best != buzz_id:
                    g.remove_node(best)
            else:
                g.add_edge(
                    foot_id,
                    best,
                    length_m=round(max(abs(g.nodes[best]["x"] - foot_x) / scale, 0.5), 3),
                    geometry=[
                        [round(foot_x, 2), round(foot_y, 2)],
                        [round(g.nodes[best]["x"], 2), round(g.nodes[best]["y"], 2)],
                    ],
                )

    # Space Mountain / Hyperspace: vertical spur up to the TL corridor (replaces long diagonal bridges).
    space = next(r for r in data["rides"] if int(r["ride_id"]) == 28)
    sx, sy = float(space["x"]), float(space["y"])
    space_id = "synthetic:space"
    space_foot = "synthetic:space_foot"
    corridor_y = 0.5 * (TL_CORRIDOR_Y_MIN + TL_CORRIDOR_Y_MAX)
    g.add_node(space_id, x=sx, y=sy, lat=None, lon=None)
    g.add_node(space_foot, x=sx, y=corridor_y, lat=None, lon=None)
    g.add_edge(
        space_id,
        space_foot,
        length_m=round(abs(sy - corridor_y) / scale, 3),
        geometry=[[round(sx, 2), round(sy, 2)], [round(sx, 2), round(corridor_y, 2)]],
    )
    # Attach foot to nearest remaining corridor node.
    corridor_nodes = {
        n: g.nodes[n]
        for n, d in g.nodes(data=True)
        if TL_CORRIDOR_X_MIN <= d["x"] <= TL_CORRIDOR_X_MAX
        and abs(d["y"] - corridor_y) < 20
        and not str(n).startswith("synthetic:space")
    }
    if corridor_nodes:
        attach = _nearest_node(corridor_nodes, sx, corridor_y)
        g.add_edge(
            space_foot,
            attach,
            length_m=round(
                max(_hypot((sx, corridor_y), (g.nodes[attach]["x"], g.nodes[attach]["y"])) / scale, 0.5),
                3,
            ),
            geometry=[
                [round(sx, 2), round(corridor_y, 2)],
                [round(g.nodes[attach]["x"], 2), round(g.nodes[attach]["y"], 2)],
            ],
        )
    space["snap_node"] = space_id
    # Drop old Space OSM snap if it only existed for the diagonal bridge.
    old_space = "1365116948"
    if old_space in g and old_space not in {r["snap_node"] for r in data["rides"]}:
        if g.degree(old_space) <= 2:
            g.remove_node(old_space)

    # Star Tours: short vertical spur south of the corridor (its OSM snap lived on the wiped south band).
    star = next(r for r in data["rides"] if int(r["ride_id"]) == 29)
    stx, sty = float(star["x"]), float(star["y"])
    star_id = "synthetic:star_tours"
    g.add_node(star_id, x=stx, y=sty, lat=None, lon=None)
    corridor_nodes = {
        n: g.nodes[n]
        for n, d in g.nodes(data=True)
        if TL_CORRIDOR_X_MIN <= d["x"] <= TL_CORRIDOR_X_MAX
        and abs(d["y"] - corridor_y) < 20
        and not str(n).startswith("synthetic:star")
    }
    if corridor_nodes:
        attach = _nearest_node(corridor_nodes, stx, corridor_y)
        # Junction on corridor at Star Tours x.
        star_foot = "synthetic:star_foot"
        g.add_node(star_foot, x=stx, y=corridor_y, lat=None, lon=None)
        g.add_edge(
            star_id,
            star_foot,
            length_m=round(abs(sty - corridor_y) / scale, 3),
            geometry=[[round(stx, 2), round(sty, 2)], [round(stx, 2), round(corridor_y, 2)]],
        )
        g.add_edge(
            star_foot,
            attach,
            length_m=round(
                max(_hypot((stx, corridor_y), (g.nodes[attach]["x"], g.nodes[attach]["y"])) / scale, 0.5),
                3,
            ),
            geometry=[
                [round(stx, 2), round(corridor_y, 2)],
                [round(g.nodes[attach]["x"], 2), round(g.nodes[attach]["y"], 2)],
            ],
        )
        star["snap_node"] = star_id


def reconnect_components(g: nx.Graph, data: dict) -> int:
    """Bridge nearby orphan components; drop far orphans with no ride/hub snaps."""
    keep = _keep_nodes(data) | {n for n in g.nodes() if str(n).startswith("synthetic:")}
    entrance = data["hubs"]["entrance"]["snap_node"]
    if entrance not in g:
        return 0
    comps = [set(c) for c in nx.connected_components(g)]
    if len(comps) <= 1:
        return 0
    main = next((c for c in comps if entrance in c), max(comps, key=len))
    scale = float(data["meta"].get("meters_to_display_scale") or 1.3155)
    max_bridge = 90.0  # display units
    added = 0
    for c in comps:
        if c == main:
            continue
        best = None
        best_d = float("inf")
        for a in c:
            ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
            for b in main:
                d = _hypot((ax, ay), (g.nodes[b]["x"], g.nodes[b]["y"]))
                if d < best_d:
                    best_d = d
                    best = (a, b)
        has_keep = bool(c & keep)
        if best is None:
            continue
        if best_d > max_bridge and not has_keep:
            g.remove_nodes_from(c)
            continue
        if best_d > max_bridge and has_keep:
            # Still bridge — required for ride reachability — but prefer closer.
            pass
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
        added += 1
    return added


def resnap_rides_and_hubs(g: nx.Graph, data: dict) -> None:
    node_coords = {n: g.nodes[n] for n in g.nodes()}
    for ride in data["rides"]:
        if str(ride.get("snap_node", "")).startswith("synthetic:"):
            # Ensure synthetic node still exists.
            if ride["snap_node"] in g:
                continue
        ride["snap_node"] = _nearest_node(node_coords, float(ride["x"]), float(ride["y"]))
    for hub in data["hubs"].values():
        hub["snap_node"] = _nearest_node(node_coords, float(hub["x"]), float(hub["y"]))


def keep_entrance_component(g: nx.Graph, data: dict) -> int:
    entrance_snap = data["hubs"]["entrance"]["snap_node"]
    synth = {n for n in g.nodes() if str(n).startswith("synthetic:")}
    if entrance_snap not in g:
        return 0
    best = None
    for comp in nx.connected_components(g):
        if entrance_snap in comp:
            best = set(comp)
            break
    if best is None:
        return 0
    # If synthetics got separated, merge-protect by skipping trim.
    if synth and not synth.issubset(best):
        return 0
    drop = set(g.nodes()) - best
    g.remove_nodes_from(drop)
    return len(drop)


def graph_to_payload(g: nx.Graph, data: dict) -> dict:
    nodes = {}
    for n, d in g.nodes(data=True):
        nodes[str(n)] = {
            "x": round(float(d["x"]), 3),
            "y": round(float(d["y"]), 3),
            "lat": d.get("lat"),
            "lon": d.get("lon"),
        }

    edges = []
    seen: set[tuple[str, str]] = set()
    for u, v, d in g.edges(data=True):
        su, sv = str(u), str(v)
        key = (su, sv) if su < sv else (sv, su)
        if key in seen:
            continue
        seen.add(key)
        geom = d.get("geometry")
        if not geom:
            geom = [
                [round(g.nodes[u]["x"], 2), round(g.nodes[u]["y"], 2)],
                [round(g.nodes[v]["x"], 2), round(g.nodes[v]["y"], 2)],
            ]
        edges.append(
            {
                "u": su,
                "v": sv,
                "length_m": round(float(d.get("length_m", 1.0)), 3),
                "geometry": geom,
            }
        )

    out = dict(data)
    out["nodes"] = nodes
    out["edges"] = edges
    out["meta"] = dict(data["meta"])
    out["meta"]["num_nodes"] = len(nodes)
    out["meta"]["num_edges"] = len(edges)
    out["meta"]["simplified"] = True
    return out


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    if data.get("meta", {}).get("simplified"):
        raise SystemExit(
            f"{PATH} is already simplified. Re-run tools/extract_osm_pathways.py first."
        )

    before_n = len(data["nodes"])
    before_e = len(data["edges"])

    nudge_rides(data)
    g = _build_graph(data)
    fix_buzz_vertical_spur(g, data)

    keep = _keep_nodes(data)
    keep |= {n for n in g.nodes() if str(n).startswith("synthetic:")}

    removed_bbox = prune_outside_bbox(g, data, keep, BBOX_PAD)
    removed_dangling = prune_dangling(g, keep)
    removed_useless = prune_useless_not_on_keep_paths(g, keep)
    removed_dangling += prune_dangling(g, keep)
    removed_spurs = prune_outward_spurs(g, keep, center=(520.0, 520.0))
    removed_dangling += prune_dangling(g, keep)

    resnap_rides_and_hubs(g, data)
    keep = _keep_nodes(data) | {n for n in g.nodes() if str(n).startswith("synthetic:")}
    removed_dangling += prune_dangling(g, keep)
    removed_spurs += prune_outward_spurs(g, keep, center=(520.0, 520.0))

    removed_comp = keep_entrance_component(g, data)
    added_bridges = reconnect_components(g, data)
    resnap_rides_and_hubs(g, data)
    keep = _keep_nodes(data) | {n for n in g.nodes() if str(n).startswith("synthetic:")}
    removed_dangling += prune_dangling(g, keep)
    removed_spurs += prune_outward_spurs(g, keep, center=(520.0, 520.0))
    removed_dangling += prune_dangling(g, keep)
    added_bridges += reconnect_components(g, data)

    if g.number_of_nodes() < 200:
        raise SystemExit(
            f"Simplification too aggressive ({g.number_of_nodes()} nodes). Aborting write."
        )
    if nx.number_connected_components(g) != 1:
        added_bridges += reconnect_components(g, data)

    out = graph_to_payload(g, data)
    PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"Simplified {PATH}: nodes {before_n}->{out['meta']['num_nodes']}, "
        f"edges {before_e}->{out['meta']['num_edges']} "
        f"(bbox={removed_bbox}, dangling={removed_dangling}, "
        f"useless={removed_useless}, spurs={removed_spurs}, "
        f"comp_drop={removed_comp}, bridges={added_bridges})"
    )
    for rid in (RISE, INDIANA, BUZZ, PIRATES, HAUNTED):
        r = next(x for x in out["rides"] if int(x["ride_id"]) == rid)
        print(f"  ride {rid} {r['name']}: ({r['x']}, {r['y']}) snap={r['snap_node']}")


if __name__ == "__main__":
    main()
