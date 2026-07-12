"""Tests for OSM pathway network integration."""

from pathways import interpolate_polyline, load_pathways, softmax_path_weights
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


def test_softmax_equal_lengths_share_mass():
    weights = softmax_path_weights([100, 100, 100], tau_sec=45.0)
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-9
    assert abs(weights[0] - weights[1]) < 1e-9
    assert abs(weights[1] - weights[2]) < 1e-9


def test_softmax_longer_path_less_likely():
    weights = softmax_path_weights([100, 145], tau_sec=45.0)
    assert weights[0] > weights[1]
    assert weights[0] > 0.7


def test_near_shortest_variants_ordered():
    net = load_pathways()
    assert net is not None
    # Entrance → Space Mountain (ride 28) should have at least one path.
    variants = net.near_shortest_variants(config.NODE_ENTRANCE, config.ride_node_id(28))
    assert len(variants) >= 1
    for a, b in zip(variants, variants[1:]):
        assert a.length_m <= b.length_m + 1e-6
    assert variants[-1].length_m <= variants[0].length_m * (1.0 + config.WALK_PATH_LENGTH_SLACK) + 1.0


def test_park_graph_variant_tables():
    graph = get_park_graph()
    assert graph.walk_variant_count.shape == (graph.num_nodes, graph.num_nodes)
    assert graph.walk_variant_base_sec.shape[2] == config.WALK_PATH_MAX_VARIANTS
    # Some OD pairs should expose alternatives when randomization is enabled.
    multi = int((graph.walk_variant_count > 1).sum())
    assert multi > 0
    i = graph.node_to_idx(graph.entrance_node)
    j = graph.node_to_idx(graph.ride_node(13))  # Matterhorn
    n = int(graph.walk_variant_count[i, j])
    assert n >= 1
    poly0 = graph.path_polyline_for_idx(i, j, variant=0)
    assert len(poly0) >= 2
    if n > 1:
        poly1 = graph.path_polyline_for_idx(i, j, variant=1)
        assert len(poly1) >= 2
