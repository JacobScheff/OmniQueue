#!/usr/bin/env python3
"""Download Disneyland walkways from OSM and write data/pathways.json.

Requires network access and optional deps: osmnx, geopandas, shapely.

Usage:
    pip install osmnx
    python tools/extract_osm_pathways.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

PLACE_NAME = "Disneyland Park, Anaheim, California, USA"
OUT_PATH = ROOT / "data" / "pathways.json"

# Display canvas (matches visualize.py defaults)
DISPLAY_WIDTH = 1000.0
DISPLAY_HEIGHT = 1000.0
DISPLAY_PAD = 40.0

# Manual lat/lon for attractions missing from OSM tourism tags.
MANUAL_COORDS: dict[str, tuple[float, float]] = {
    "Sailing Ship Columbia": (33.81235, -117.92055),
    "Mark Twain Riverboat": (33.81215, -117.92085),
}

# OSM name aliases → config ride name
RIDE_ALIASES: dict[str, list[str]] = {
    "Star Wars: Rise of the Resistance": ["Star Wars: Rise of the Resistance"],
    "Millennium Falcon: Smugglers Run": [
        "Millennium Falcon: Smuggler’s Run",
        "Millennium Falcon: Smugglers Run",
    ],
    "Tiana's Bayou Adventure": ["Tiana's Bayou Adventure"],
    "The Many Adventures of Winnie the Pooh": ["The Many Adventures of Winnie the Pooh"],
    "Davy Crockett's Explorer Canoes": [
        "Davy Crockett's Explorer Canoes",
        "Davy Crockett Explorer Canoes",
    ],
    "Haunted Mansion": ["The Haunted Mansion", "Haunted Mansion"],
    "Pirates of the Caribbean": ["Pirates of the Caribbean"],
    "Indiana Jones Adventure": ["Indiana Jones Adventure"],
    "Jungle Cruise": ["Jungle Cruise"],
    "Walt Disney's Enchanted Tiki Room": ["Walt Disney's Enchanted Tiki Room"],
    "Big Thunder Mountain Railroad": ["Big Thunder Mountain Railroad"],
    "Mark Twain Riverboat": ["Mark Twain Riverboat"],
    "Sailing Ship Columbia": ["Sailing Ship Columbia"],
    "Matterhorn Bobsleds": ["Matterhorn Bobsleds"],
    "Peter Pan's Flight": ["Peter Pan's Flight"],
    "Mr. Toad's Wild Ride": ["Mr. Toad's Wild Ride"],
    "Snow White's Enchanted Wish": ["Snow White's Enchanted Wish"],
    "Pinocchio's Daring Journey": ["Pinocchio's Daring Journey"],
    "King Arthur Carrousel": ["King Arthur Carrousel"],
    "Dumbo the Flying Elephant": ["Dumbo the Flying Elephant"],
    "Mad Tea Party": ["Mad Tea Party"],
    "Alice in Wonderland": ["Alice in Wonderland"],
    "Casey Jr. Circus Train": ["Casey Jr. Circus Train"],
    "Storybook Land Canal Boats": ["Storybook Land Canal Boats"],
    "it's a small world": [
        "\"it's a small world\"",
        "it's a small world",
        "It's a Small World",
        "\"It's a Small World\"",
    ],
    "Mickey and Minnie's Runaway Railway": ["Mickey and Minnie's Runaway Railway"],
    "Roger Rabbit's Car Toon Spin": ["Roger Rabbit's Car Toon Spin"],
    "Chip 'n' Dale's GADGETcoaster": [
        "Chip ‘n’ Dale’s GADGETcoaster",
        "Chip 'n' Dale's GADGETcoaster",
    ],
    "Space Mountain": ["Space Mountain", "Hyperspace Mountain"],
    "Star Tours": ["Star Tours: The Adventures Continue", "Star Tours"],
    "Buzz Lightyear Astro Blasters": ["Buzz Lightyear Astro Blasters"],
    "Astro Orbitor": ["Astro Orbitor"],
    "Autopia": ["Autopia"],
    "Finding Nemo Submarine Voyage": ["Finding Nemo Submarine Voyage"],
    "Disneyland Monorail": ["Tomorrowland Monorail Station", "Disneyland Monorail"],
}

# Land hub anchors as (lat, lon) — used when OSM landmarks exist, else ride centroids.
HUB_LANDMARKS: dict[int, list[str]] = {
    config.NODE_MAIN_HUB: ["Partners"],
    config.NODE_CENTRAL_PLAZA: ["Sleeping Beauty Castle"],
    config.NODE_ENTRANCE: ["Town Square"],
}


def _mean_latlon(pts: list[tuple[float, float]]) -> tuple[float, float]:
    lat = sum(p[0] for p in pts) / len(pts)
    lon = sum(p[1] for p in pts) / len(pts)
    return lat, lon


def _project_factory(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float
):
    """Map WGS84 → park display coords; also report meters-per-degree scales."""
    lat0 = 0.5 * (lat_min + lat_max)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    usable_w = DISPLAY_WIDTH - 2 * DISPLAY_PAD
    usable_h = DISPLAY_HEIGHT - 2 * DISPLAY_PAD
    width_m = (lon_max - lon_min) * m_per_deg_lon
    height_m = (lat_max - lat_min) * m_per_deg_lat
    scale = min(usable_w / max(width_m, 1e-6), usable_h / max(height_m, 1e-6))

    # Center the park in the canvas.
    offset_x = DISPLAY_PAD + 0.5 * (usable_w - width_m * scale)
    offset_y = DISPLAY_PAD + 0.5 * (usable_h - height_m * scale)

    def project(lat: float, lon: float) -> tuple[float, float]:
        east_m = (lon - lon_min) * m_per_deg_lon
        south_m = (lat_max - lat) * m_per_deg_lat  # Y increases south (pygame)
        return offset_x + east_m * scale, offset_y + south_m * scale

    return project, scale, m_per_deg_lat, m_per_deg_lon


def _simplify_coords(coords: list[list[float]], tol: float = 1.5) -> list[list[float]]:
    """Drop near-colinear intermediate points (display units)."""
    if len(coords) <= 2:
        return coords
    keep = [coords[0]]
    for i in range(1, len(coords) - 1):
        ax, ay = keep[-1]
        bx, by = coords[i]
        cx, cy = coords[i + 1]
        # Perpendicular distance from B to AC
        dx, dy = cx - ax, cy - ay
        denom = math.hypot(dx, dy) or 1.0
        dist = abs((bx - ax) * dy - (by - ay) * dx) / denom
        if dist >= tol:
            keep.append(coords[i])
    keep.append(coords[-1])
    return keep


def main() -> None:
    try:
        import networkx as nx
        import osmnx as ox
    except ImportError as exc:
        raise SystemExit(
            "osmnx is required. Install with: pip install osmnx\n" + str(exc)
        ) from exc

    print(f"Downloading walk network for {PLACE_NAME!r}...", flush=True)
    G = ox.graph_from_place(PLACE_NAME, network_type="walk")
    UG = ox.convert.to_undirected(G)
    print(f"  nodes={UG.number_of_nodes()} edges={UG.number_of_edges()}")

    print("Downloading attraction features...", flush=True)
    feats = ox.features_from_place(
        PLACE_NAME, tags={"tourism": True, "attraction": True}
    )

    centroids: dict[str, list[tuple[float, float]]] = {}
    for _, row in feats.iterrows():
        name = row.get("name")
        if not isinstance(name, str):
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        centroids.setdefault(name, []).append((float(c.y), float(c.x)))

    # Bounds from walk graph
    lons = [float(d["x"]) for _, d in UG.nodes(data=True)]
    lats = [float(d["y"]) for _, d in UG.nodes(data=True)]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    project, scale, m_per_deg_lat, m_per_deg_lon = _project_factory(
        lon_min, lon_max, lat_min, lat_max
    )

    # Resolve ride lat/lon
    ride_latlon: list[tuple[float, float]] = []
    ride_sources: list[str] = []
    for ride in config.RIDES:
        name = ride["name"]
        found = None
        source = "manual"
        for alias in RIDE_ALIASES.get(name, [name]):
            if alias in centroids:
                found = _mean_latlon(centroids[alias])
                source = f"osm:{alias}"
                break
        if found is None and name in MANUAL_COORDS:
            found = MANUAL_COORDS[name]
            source = "manual"
        if found is None:
            raise SystemExit(f"No coordinates for ride {name!r}")
        ride_latlon.append(found)
        ride_sources.append(source)
        print(f"  ride {name}: {found} ({source})")

    # Hub lat/lon from landmarks or mean of assigned rides
    hub_latlon: dict[int, tuple[float, float]] = {}
    for hub_id, names in HUB_LANDMARKS.items():
        for n in names:
            if n in centroids:
                hub_latlon[hub_id] = _mean_latlon(centroids[n])
                break

    for hub_id in config.HUB_COORDS:
        if hub_id in hub_latlon:
            continue
        if hub_id == config.NODE_ENTRANCE:
            # South edge center near Town Square / Main Street
            hub_latlon[hub_id] = (lat_min + 0.00015, 0.5 * (lon_min + lon_max))
            continue
        # Average rides assigned to this hub
        pts = [
            ride_latlon[rid]
            for rid, h in enumerate(config.RIDE_HUB)
            if h == hub_id
        ]
        if hub_id == config.NODE_RIVER_CROSSING:
            pts = [ride_latlon[10], ride_latlon[11], ride_latlon[5]]  # BTMR, Twain, HM
        if hub_id == config.NODE_CENTRAL_PLAZA and hub_id not in hub_latlon:
            if "Sleeping Beauty Castle" in centroids:
                hub_latlon[hub_id] = _mean_latlon(centroids["Sleeping Beauty Castle"])
            elif "Partners" in centroids:
                hub_latlon[hub_id] = _mean_latlon(centroids["Partners"])
        if pts and hub_id not in hub_latlon:
            hub_latlon[hub_id] = _mean_latlon(pts)

    # Project pathway nodes
    nodes_out: dict[str, dict] = {}
    for nid, data in UG.nodes(data=True):
        lat, lon = float(data["y"]), float(data["x"])
        x, y = project(lat, lon)
        nodes_out[str(nid)] = {
            "x": round(x, 3),
            "y": round(y, 3),
            "lat": lat,
            "lon": lon,
        }

    def nearest_node(lat: float, lon: float) -> str:
        best, best_d = None, float("inf")
        for nid, data in UG.nodes(data=True):
            dlat = (float(data["y"]) - lat) * m_per_deg_lat
            dlon = (float(data["x"]) - lon) * m_per_deg_lon
            d = dlat * dlat + dlon * dlon
            if d < best_d:
                best_d = d
                best = nid
        return str(best)

    edges_out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for u, v, data in UG.edges(data=True):
        su, sv = str(u), str(v)
        key = (su, sv) if su < sv else (sv, su)
        if key in seen:
            continue
        seen.add(key)
        length_m = float(data.get("length", 0.0))
        geom = data.get("geometry")
        if geom is not None:
            # Shapely coords are (lon, lat)
            coords_ll = [(float(lon), float(lat)) for lon, lat in geom.coords]
        else:
            coords_ll = [
                (float(UG.nodes[u]["x"]), float(UG.nodes[u]["y"])),
                (float(UG.nodes[v]["x"]), float(UG.nodes[v]["y"])),
            ]
        poly = []
        for lon, lat in coords_ll:
            px, py = project(lat, lon)
            poly.append([round(px, 2), round(py, 2)])
        # Ensure endpoints match node projection orientation u→v
        if poly and (
            abs(poly[0][0] - nodes_out[su]["x"]) + abs(poly[0][1] - nodes_out[su]["y"])
            > abs(poly[-1][0] - nodes_out[su]["x"]) + abs(poly[-1][1] - nodes_out[su]["y"])
        ):
            poly = list(reversed(poly))
        poly = _simplify_coords(poly)
        edges_out.append(
            {
                "u": su,
                "v": sv,
                "length_m": round(length_m, 3),
                "geometry": poly,
            }
        )

    rides_out = []
    for rid, ride in enumerate(config.RIDES):
        lat, lon = ride_latlon[rid]
        x, y = project(lat, lon)
        snap = nearest_node(lat, lon)
        rides_out.append(
            {
                "ride_id": rid,
                "name": ride["name"],
                "x": round(x, 3),
                "y": round(y, 3),
                "lat": lat,
                "lon": lon,
                "snap_node": snap,
                "source": ride_sources[rid],
            }
        )

    hubs_out = {}
    hub_names = {
        config.NODE_ENTRANCE: "entrance",
        config.NODE_MAIN_HUB: "main_hub",
        config.NODE_GALAXY_HUB: "galaxy_hub",
        config.NODE_CRITTER_HUB: "critter_hub",
        config.NODE_NEW_ORLEANS_HUB: "new_orleans_hub",
        config.NODE_ADVENTURE_HUB: "adventure_hub",
        config.NODE_FRONTIER_HUB: "frontier_hub",
        config.NODE_FANTASY_HUB: "fantasy_hub",
        config.NODE_TOONTOWN_HUB: "toontown_hub",
        config.NODE_TOMORROW_HUB: "tomorrow_hub",
        config.NODE_RIVER_CROSSING: "river_crossing",
        config.NODE_CENTRAL_PLAZA: "central_plaza",
    }
    for hub_id, key in hub_names.items():
        lat, lon = hub_latlon[hub_id]
        x, y = project(lat, lon)
        hubs_out[key] = {
            "node_id": hub_id,
            "x": round(x, 3),
            "y": round(y, 3),
            "lat": lat,
            "lon": lon,
            "snap_node": nearest_node(lat, lon),
        }

    payload = {
        "meta": {
            "place": PLACE_NAME,
            "source": "OpenStreetMap via osmnx",
            "network_type": "walk",
            "display_width": DISPLAY_WIDTH,
            "display_height": DISPLAY_HEIGHT,
            "display_pad": DISPLAY_PAD,
            "meters_to_display_scale": scale,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "num_nodes": len(nodes_out),
            "num_edges": len(edges_out),
        },
        "nodes": nodes_out,
        "edges": edges_out,
        "rides": rides_out,
        "hubs": hubs_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KiB)")
    print(
        f"  {payload['meta']['num_nodes']} nodes, "
        f"{payload['meta']['num_edges']} edges, "
        f"scale={scale:.4f} display-units/m"
    )

    # Light cleanup + ride nudges (Indiana / Rise / Buzz).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from simplify_pathways import main as simplify_main

    simplify_main()


if __name__ == "__main__":
    main()
