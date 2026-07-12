"""Tests for OSM pathway network integration."""

from pathways import interpolate_polyline, load_pathways
from park_graph import get_park_graph
import config


def test_pathways_json_loads():
    net = load_pathways()
    assert net is not None
    assert len(net.nodes) > 100
    assert len(net.edges_raw) > 100
    assert len(net.rides) == config.NUM_RIDES


def test_pathway_coords_applied_to_config():
    get_park_graph()  # applies pathway overlay
    net = load_pathways()
    assert net is not None
    # Entrance should be near the south of the canvas after projection.
    ex, ey = config.ENTRANCE_COORDS
    assert ey > 700
    # Galaxy's Edge / Rise should be northwest (low x, relatively low y).
    rx, ry = config.RIDES[0]["coords"]
    assert rx < 400
    assert ry < 500


def test_walk_times_use_pathway_meters():
    graph = get_park_graph()
    entrance = graph.entrance_node
    # Rise of the Resistance is far from the entrance along walkways.
    far = graph.walk_time(entrance, graph.ride_node(0))
    # Astro Orbitor is much closer to the hub / entrance corridor.
    nearish = graph.walk_time(entrance, graph.ride_node(31))
    assert far > 60
    assert nearish > 0
    # Pathway route should be longer than a naive straight-line estimate in many cases;
    # at minimum both rides are reachable.
    assert far != nearish


def test_path_polyline_curves():
    graph = get_park_graph()
    i = graph.node_to_idx(graph.entrance_node)
    j = graph.node_to_idx(graph.ride_node(0))
    poly = graph.path_polyline_for_idx(i, j)
    assert len(poly) >= 3  # not a straight hub spoke


def test_interpolate_polyline_endpoints():
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert interpolate_polyline(pts, 0.0) == (0.0, 0.0)
    assert interpolate_polyline(pts, 1.0) == (10.0, 10.0)
    mid = interpolate_polyline(pts, 0.5)
    assert abs(mid[0] - 10.0) < 1e-6
    assert abs(mid[1] - 0.0) < 1e-6
