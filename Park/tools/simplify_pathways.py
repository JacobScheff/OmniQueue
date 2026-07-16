#!/usr/bin/env python3
"""Post-process data/pathways.json: ride placements + path cleanup.

Layout polish (layout-v4):
- One straight vertical Main Street from entrance to the hub circle
- Pirates on the Adventureland 3-path vertex (spur removed)
- Haunted Mansion into the dense New Orleans path cluster (old spur removed)
- Small World at the top of its plaza path collection
- Keep the western Critter↔GE round corridor; drop nowhere scribbles left of Rise
- Space Mountain / Indiana / Autopia / Rise / Buzz spur (unchanged intent)

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
FALCON = 1
TIANA = 2
POOH = 3
DAVY = 4
HAUNTED = 5
PIRATES = 6
INDIANA = 7
JUNGLE = 8
SMALL_WORLD = 24
SPACE = 28
BUZZ = 30
AUTOPIA = 32
MATTERHORN = 13
NEMO = 33

TL_CORRIDOR_Y_MIN = 560.0
TL_CORRIDOR_Y_MAX = 585.0
TL_CORRIDOR_X_MIN = 640.0
TL_CORRIDOR_X_MAX = 820.0

BUZZ_ONLY_LOOP_X_MIN = 750.0
BUZZ_ONLY_LOOP_X_MAX = 805.0
BUZZ_ONLY_LOOP_Y_MIN = 495.0
BUZZ_ONLY_LOOP_Y_MAX = 550.0

# Main Street band (entrance south → hub). Y increases toward the entrance.
MS_X_PAD = 55.0


def _hypot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_node(nodes: dict, x: float, y: float, exclude: set[str] | None = None) -> str:
    best, best_d = None, float("inf")
    for nid, nd in nodes.items():
        if exclude and nid in exclude:
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


def _leaf_nodes(g: nx.Graph) -> list[str]:
    return [n for n in g.nodes() if g.degree(n) == 1]


def _protect_set(g: nx.Graph, data: dict) -> set[str]:
    keep = {str(r["snap_node"]) for r in data["rides"] if r.get("snap_node") is not None}
    keep |= {str(h["snap_node"]) for h in data["hubs"].values()}
    keep |= {n for n in g.nodes() if str(n).startswith("synthetic:")}
    return keep


def _add_edge(g: nx.Graph, a: str, b: str, scale: float) -> None:
    if a == b or g.has_edge(a, b):
        return
    ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
    bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
    g.add_edge(
        a,
        b,
        length_m=round(max(_hypot((ax, ay), (bx, by)) / scale, 0.5), 3),
        geometry=[[round(ax, 2), round(ay, 2)], [round(bx, 2), round(by, 2)]],
    )


def _trim_spur_to_junction(g: nx.Graph, leaf: str, keep: set[str]) -> int:
    """Remove a degree-1 chain up to (but not including) the first remaining branch.

    After deleting the leaf, a former 3-way junction has degree 2 — stop there so we
    do not cascade into the main corridor.
    """
    if leaf not in g or g.degree(leaf) != 1:
        return 0
    removed = 0
    cur, prev = leaf, None
    while cur in g and g.degree(cur) <= 2 and cur not in keep:
        nbrs = [n for n in g.neighbors(cur) if n != prev]
        nxt = nbrs[0] if nbrs else None
        g.remove_node(cur)
        removed += 1
        if nxt is None:
            break
        # Former junction (>=3) now has deg >= 2; chain node now has deg 1.
        if g.degree(nxt) >= 2:
            break
        prev, cur = cur, nxt
    return removed


def nudge_rides(g: nx.Graph, data: dict) -> None:
    rides = {int(r["ride_id"]): r for r in data["rides"]}

    # --- Space Mountain: tip of the stub almost directly above the old OSM spot ---
    sx = float(rides[SPACE]["x"])
    sy = float(rides[SPACE]["y"])
    above = []
    for n in _leaf_nodes(g):
        d = g.nodes[n]
        if sy - 120 <= d["y"] <= sy - 20 and abs(d["x"] - sx) < 25:
            above.append((abs(d["x"] - sx), -d["y"], n, d["x"], d["y"]))
    above.sort()
    if above:
        near = [t for t in above if t[0] < 12.0] or above
        near.sort(key=lambda t: t[1])
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
        if d["y"] > jy + 15 and jx - 120 <= d["x"] <= jx + 20:
            south.append((d["y"], abs(d["x"] - (jx - 40)), n, d["x"], d["y"]))
    south.sort(reverse=True)
    if south:
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

    # --- Pirates: at the 3-path vertex; drop the approach spur ---
    px = float(rides[PIRATES]["x"])
    py = float(rides[PIRATES]["y"])
    junctions = []
    for n, d in g.nodes(data=True):
        if g.degree(n) < 3:
            continue
        if 220 <= d["x"] <= 310 and 700 <= d["y"] <= 760:
            junctions.append((_hypot((d["x"], d["y"]), (px, py)), n, d["x"], d["y"]))
    junctions.sort()
    if junctions:
        _, node, x, y = junctions[0]
        # Prefer the clearest Y near the pirates leaf if present.
        pirates_snap = str(rides[PIRATES].get("snap_node") or "")
        if pirates_snap in g and g.degree(pirates_snap) == 1:
            nbr = next(iter(g.neighbors(pirates_snap)))
            if g.degree(nbr) >= 3:
                node = nbr
                x, y = g.nodes[nbr]["x"], g.nodes[nbr]["y"]
        old_snap = pirates_snap
        rides[PIRATES]["x"] = round(x, 3)
        rides[PIRATES]["y"] = round(y, 3)
        rides[PIRATES]["snap_node"] = node
        rides[PIRATES]["source"] = "simplified:y-junction"
        keep = _protect_set(g, data) | {node}
        if old_snap in g and old_snap != node:
            _trim_spur_to_junction(g, old_snap, keep)

    # --- Haunted Mansion: dense New Orleans path tangle (right / slightly up of OSM) ---
    # Absolute cluster (display): east of the old western spur, around (200–220, 610–640).
    best, best_score = None, -1.0
    for n, d in g.nodes(data=True):
        if not (175 <= d["x"] <= 230 and 600 <= d["y"] <= 650):
            continue
        nearby = sum(
            1
            for _, d2 in g.nodes(data=True)
            if _hypot((d["x"], d["y"]), (d2["x"], d2["y"])) < 30
        )
        score = nearby * 2.0 + g.degree(n) * 3.0
        if score > best_score:
            best_score = score
            best = (n, d["x"], d["y"])
    if best:
        old_snap = str(rides[HAUNTED].get("snap_node") or "")
        n, x, y = best
        rides[HAUNTED]["x"] = round(x, 3)
        rides[HAUNTED]["y"] = round(y, 3)
        rides[HAUNTED]["snap_node"] = n
        rides[HAUNTED]["source"] = "simplified:nos-cluster"
        keep = _protect_set(g, data) | {n}
        # Protect Critter Country corridor so spur trim cannot eat the HM↔Critter link.
        for rid in (TIANA, POOH, DAVY):
            snap = str(rides[rid]["snap_node"])
            if snap in g:
                try:
                    keep.update(nx.shortest_path(g, n, snap, weight="length_m"))
                except nx.NetworkXNoPath:
                    pass
        if old_snap in g and old_snap != n and g.degree(old_snap) == 1:
            _trim_spur_to_junction(g, old_snap, keep)
        # Only drop leftover dead-end leaves west of HM that are not on Critter paths.
        for leaf in list(_leaf_nodes(g)):
            d = g.nodes[leaf]
            if d["x"] < x - 8 and 600 <= d["y"] <= 670 and leaf not in keep:
                _trim_spur_to_junction(g, leaf, keep)

    # --- Pooh: pull onto the dense Critter Country path bundle near Tiana/Davy ---
    bundle = [
        (n, d["x"], d["y"])
        for n, d in g.nodes(data=True)
        if 110 <= d["x"] <= 175 and 500 <= d["y"] <= 545 and g.degree(n) >= 2
    ]
    if bundle:
        best, best_score = None, -1.0
        for n, x, y in bundle:
            nearby = sum(1 for _, x2, y2 in bundle if _hypot((x, y), (x2, y2)) < 28)
            # Prefer denser nodes on the west/north edge of the Davy–Tiana tangle.
            score = nearby + 0.5 * g.degree(n) - 0.02 * abs(x - 140) - 0.01 * abs(y - 525)
            if score > best_score:
                best_score = score
                best = (n, x, y)
        if best:
            old_snap = str(rides[POOH].get("snap_node") or "")
            n, x, y = best
            rides[POOH]["x"] = round(x, 3)
            rides[POOH]["y"] = round(y, 3)
            rides[POOH]["snap_node"] = n
            rides[POOH]["source"] = "simplified:critter-bundle"
            keep = _protect_set(g, data) | {n}
            if old_snap in g and old_snap != n and g.degree(old_snap) == 1:
                _trim_spur_to_junction(g, old_snap, keep)

    # --- Small World: top (lowest y) of the plaza path collection above it ---
    swx, swy = float(rides[SMALL_WORLD]["x"]), float(rides[SMALL_WORLD]["y"])
    cluster = [
        (n, d["x"], d["y"])
        for n, d in g.nodes(data=True)
        if 720 <= d["x"] <= 780 and 200 <= d["y"] <= 300
    ]
    if cluster:
        top_y = min(y for _, _, y in cluster)
        # Prefer the highest row of the eastern plaza tangle (near Small World's lon).
        top = [
            (y, -g.degree(n), abs(x - 752.0), n, x, y)
            for n, x, y in cluster
            if y <= top_y + 8 and g.degree(n) >= 2
        ]
        top.sort()
        if not top:
            top = [
                (y, -g.degree(n), abs(x - 752.0), n, x, y)
                for n, x, y in cluster
                if y <= top_y + 12
            ]
            top.sort()
        if top:
            _, _, _, n, x, y = top[0]
            rides[SMALL_WORLD]["x"] = round(x, 3)
            rides[SMALL_WORLD]["y"] = round(y, 3)
            rides[SMALL_WORLD]["snap_node"] = n
            rides[SMALL_WORLD]["source"] = "simplified:plaza-top"

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
    for rid in (MATTERHORN, 29, 31, AUTOPIA, NEMO, SPACE):
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


def straighten_main_street(g: nx.Graph, data: dict) -> int:
    """Replace the Main Street corridor with one straight vertical polyline."""
    scale = float(data["meta"].get("meters_to_display_scale") or 1.3155)
    entrance = data["hubs"]["entrance"]
    hub = data["hubs"]["main_hub"]
    ent_id = str(entrance["snap_node"])
    hub_id = str(hub["snap_node"])
    if ent_id not in g or hub_id not in g:
        return 0

    hub_x = float(hub["x"])
    # Prefer the geometric center of the hub circle if nearby ring nodes exist.
    ring = [
        d["x"]
        for n, d in g.nodes(data=True)
        if abs(d["y"] - float(hub["y"])) < 12 and abs(d["x"] - hub_x) < 40
    ]
    if ring:
        hub_x = sum(ring) / len(ring)

    ent_y = float(g.nodes[ent_id]["y"])
    hub_y = float(g.nodes[hub_id]["y"])
    y_lo, y_hi = min(hub_y, ent_y), max(hub_y, ent_y)

    # Side exits that leave the MS band (must be reattached to the straight line).
    ms_nodes = {
        n
        for n, d in g.nodes(data=True)
        if y_lo + 8 <= d["y"] <= y_hi - 8 and abs(d["x"] - hub_x) <= MS_X_PAD
    }
    # Keep entrance/hub; do not delete ride snaps.
    keep = _protect_set(g, data)
    ms_nodes -= keep
    ms_nodes.discard(ent_id)
    ms_nodes.discard(hub_id)

    side_reattach: list[str] = []
    for n in list(ms_nodes):
        if n not in g:
            continue
        for nbr in list(g.neighbors(n)):
            if nbr in ms_nodes or nbr in {ent_id, hub_id}:
                continue
            if nbr not in keep and abs(g.nodes[nbr]["x"] - hub_x) <= MS_X_PAD:
                # Still inside the wider MS corridor — will be deleted if in ms_nodes.
                continue
            side_reattach.append(nbr)

    removed = 0
    for n in list(ms_nodes):
        if n in g:
            g.remove_node(n)
            removed += 1

    # Also purge leftover MS wobble nodes close to the axis that are not protected.
    for n, d in list(g.nodes(data=True)):
        if n in keep or n in {ent_id, hub_id} or str(n).startswith("synthetic:"):
            continue
        if y_lo + 5 <= d["y"] <= y_hi - 5 and abs(d["x"] - hub_x) <= 22:
            # Only drop if it has no outside connections beyond MS.
            outside = [
                nbr
                for nbr in g.neighbors(n)
                if abs(g.nodes[nbr]["x"] - hub_x) > 22
                or not (y_lo <= g.nodes[nbr]["y"] <= y_hi)
            ]
            if not outside:
                for nbr in list(g.neighbors(n)):
                    if nbr not in keep and nbr not in {ent_id, hub_id}:
                        side_reattach.append(nbr)
                g.remove_node(n)
                removed += 1

    # Build straight chain hub → entrance.
    step = 22.0
    ys: list[float] = []
    y = hub_y + (step if ent_y > hub_y else -step)
    if ent_y > hub_y:
        while y < ent_y - 8:
            ys.append(y)
            y += step
    else:
        while y > ent_y + 8:
            ys.append(y)
            y -= step

    chain = [hub_id]
    for i, yy in enumerate(ys):
        nid = f"synthetic:ms_{i}"
        g.add_node(nid, x=round(hub_x, 3), y=round(yy, 3), lat=None, lon=None)
        chain.append(nid)
    chain.append(ent_id)

    # Snap entrance onto the vertical axis.
    g.nodes[ent_id]["x"] = round(hub_x, 3)
    entrance["x"] = round(hub_x, 3)
    hub["x"] = round(hub_x, 3)
    g.nodes[hub_id]["x"] = round(hub_x, 3)

    for a, b in zip(chain, chain[1:]):
        # Drop any old edges between endpoints that zigzag.
        if g.has_edge(a, b):
            g.remove_edge(a, b)
        _add_edge(g, a, b, scale)

    # Remove non-chain edges between chain nodes (keep the single polyline).
    chain_set = set(chain)
    chain_index = {n: i for i, n in enumerate(chain)}
    for a in chain:
        idx = chain_index[a]
        allowed = set()
        if idx > 0:
            allowed.add(chain[idx - 1])
        if idx + 1 < len(chain):
            allowed.add(chain[idx + 1])
        for nbr in list(g.neighbors(a)):
            if nbr in chain_set and nbr not in allowed:
                g.remove_edge(a, nbr)

    # Reattach side paths to the nearest point on the straight line.
    for nbr in dict.fromkeys(side_reattach):
        if nbr not in g:
            continue
        best, best_d = None, float("inf")
        for c in chain:
            d = abs(g.nodes[c]["y"] - g.nodes[nbr]["y"])
            if d < best_d:
                best_d = d
                best = c
        if best is not None:
            _add_edge(g, nbr, best, scale)

    return removed


def prune_rise_left_scribbles(g: nx.Graph, data: dict) -> int:
    """Remove nowhere scribbles left of Rise; keep Critter↔GE western corridor."""
    rides = {int(r["ride_id"]): r for r in data["rides"]}
    scale = float(data["meta"].get("meters_to_display_scale") or 1.3155)
    rx = float(rides[RISE]["x"])
    ry = float(rides[RISE]["y"])
    keep = _protect_set(g, data)

    # Snap Rise onto the GE network just south of the NW stub (not a west leaf).
    ge_nodes = [
        (abs(d["x"] - 120.0) + 0.3 * abs(d["y"] - 290.0), n, d["x"], d["y"])
        for n, d in g.nodes(data=True)
        if 95 <= d["x"] <= 140 and 270 <= d["y"] <= 320 and g.degree(n) >= 2
    ]
    ge_nodes.sort()
    if ge_nodes:
        old = str(rides[RISE].get("snap_node") or "")
        _, node, x, y = ge_nodes[0]
        rides[RISE]["x"] = round(x, 3)
        rides[RISE]["y"] = round(y, 3)
        rides[RISE]["snap_node"] = node
        rides[RISE]["source"] = "simplified:northwest-ge"
        rx, ry = x, y
        keep.add(node)
        if old in g and old != node and g.degree(old) == 1:
            _trim_spur_to_junction(g, old, keep | {node})

    critter = []
    ge = []
    for rid in (TIANA, POOH, DAVY):
        snap = str(rides[rid]["snap_node"])
        if snap in g:
            critter.append(snap)
            keep.add(snap)
    for rid in (RISE, FALCON):
        snap = str(rides[rid]["snap_node"])
        if snap in g:
            ge.append(snap)
            keep.add(snap)
    for a in critter:
        for b in ge:
            try:
                keep.update(nx.shortest_path(g, a, b, weight="length_m"))
            except nx.NetworkXNoPath:
                pass

    removed = 0
    # Tight box immediately west of Rise (not the southern Critter corridor).
    for n, d in list(g.nodes(data=True)):
        if n in keep:
            continue
        if d["x"] < rx - 8 and ry - 50 <= d["y"] <= ry + 25:
            g.remove_node(n)
            removed += 1

    # Peel dangling west stubs (leaves) while preserving corridor loops.
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes()):
            if n in keep or str(n).startswith("synthetic:"):
                continue
            d = g.nodes[n]
            if d["x"] >= 155 or d["y"] >= 480:
                continue
            if g.degree(n) <= 1:
                g.remove_node(n)
                removed += 1
                changed = True

    # Flatten edge geometries that poke west of Rise near GE (nowhere scribble look).
    for u, v, ed in g.edges(data=True):
        geom = ed.get("geometry") or []
        if len(geom) < 3:
            continue
        # Only touch edges near Rise latitude.
        ys = [float(p[1]) for p in geom]
        xs = [float(p[0]) for p in geom]
        if max(ys) < ry - 40 or min(ys) > ry + 40:
            continue
        if min(xs) >= rx - 5:
            continue
        # If this edge is on the protected Critter↔GE corridor below Rise, keep detail
        # when both endpoints are south of Rise.
        if g.nodes[u]["y"] > ry + 30 and g.nodes[v]["y"] > ry + 30:
            continue
        ed["geometry"] = [
            [round(g.nodes[u]["x"], 2), round(g.nodes[u]["y"], 2)],
            [round(g.nodes[v]["x"], 2), round(g.nodes[v]["y"], 2)],
        ]
        ed["length_m"] = round(
            max(_hypot((g.nodes[u]["x"], g.nodes[u]["y"]), (g.nodes[v]["x"], g.nodes[v]["y"])) / scale, 0.5),
            3,
        )
    return removed


def prune_main_street_side_stubs(g: nx.Graph, data: dict) -> int:
    """Drop short dead-end stubs sticking out of the straight Main Street."""
    keep = _protect_set(g, data)
    hub = data["hubs"]["main_hub"]
    hub_x = float(hub["x"])
    hub_y = float(g.nodes[str(hub["snap_node"])]["y"]) if str(hub["snap_node"]) in g else float(hub["y"])
    ent = data["hubs"]["entrance"]
    ent_y = float(g.nodes[str(ent["snap_node"])]["y"]) if str(ent["snap_node"]) in g else float(ent["y"])
    y_lo, y_hi = min(hub_y, ent_y), max(hub_y, ent_y)
    removed = 0
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes()):
            if n in keep or str(n).startswith("synthetic:"):
                continue
            d = g.nodes[n]
            if not (y_lo - 5 <= d["y"] <= y_hi + 5):
                continue
            # Stubs off to the side of the vertical.
            if abs(d["x"] - hub_x) < 8:
                continue
            if 500 <= d["x"] <= 660 and g.degree(n) <= 1:
                g.remove_node(n)
                removed += 1
                changed = True
    return removed


def lightly_simplify_small_world_cluster(g: nx.Graph, data: dict) -> int:
    """Slightly thin the dense plaza where Small World sits."""
    rides = {int(r["ride_id"]): r for r in data["rides"]}
    sw = rides[SMALL_WORLD]
    cx, cy = float(sw["x"]), float(sw["y"])
    keep = _protect_set(g, data)

    cluster = [
        n
        for n, d in g.nodes(data=True)
        if _hypot((d["x"], d["y"]), (cx, cy)) < 70
    ]
    removed = 0
    deg2 = [n for n in cluster if n not in keep and g.degree(n) == 2]

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
    out["meta"]["simplify_level"] = "layout-v4"
    return out


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    if data.get("meta", {}).get("simplified"):
        raise SystemExit(
            f"{PATH} is already simplified. Re-run tools/extract_osm_pathways.py first."
        )

    for r in data["rides"]:
        if r.get("name") == "Hyperspace Mountain":
            r["name"] = "Space Mountain"

    before_n, before_e = len(data["nodes"]), len(data["edges"])
    g = _build_graph(data)
    nudge_rides(g, data)
    fix_buzz_vertical_spur(g, data)

    ms_removed = straighten_main_street(g, data)
    ms_stubs = prune_main_street_side_stubs(g, data)
    west_removed = prune_rise_left_scribbles(g, data)
    sw_removed = lightly_simplify_small_world_cluster(g, data)

    resnap_rides_and_hubs(g, data)

    coords = {n: g.nodes[n] for n in g.nodes()}
    for ride in data["rides"]:
        if ride["snap_node"] not in g:
            ride["snap_node"] = _nearest_node(coords, float(ride["x"]), float(ride["y"]))

    if nx.number_connected_components(g) != 1:
        entrance = data["hubs"]["entrance"]["snap_node"]
        keep = _protect_set(g, data)
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
                _add_edge(g, a, b, scale)
                main |= c

    out = graph_to_payload(g, data)
    PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"Simplified {PATH}: nodes {before_n}->{out['meta']['num_nodes']}, "
        f"edges {before_e}->{out['meta']['num_edges']} "
        f"(ms={ms_removed}+stubs{ms_stubs}, west={west_removed}, sw={sw_removed})"
    )
    for rid in (RISE, POOH, HAUNTED, PIRATES, INDIANA, SMALL_WORLD, SPACE, BUZZ, AUTOPIA):
        r = next(x for x in out["rides"] if int(x["ride_id"]) == rid)
        print(f"  ride {rid} {r['name']}: ({r['x']}, {r['y']}) snap={r['snap_node']}")


if __name__ == "__main__":
    main()
