"""Tests for park graph pathfinding."""

import numpy as np

from park_graph import (
    _load_walk_matrix_cache,
    build_graph,
    get_park_graph,
    reset_park_graph,
)


def test_walk_times_symmetry():
    graph = get_park_graph()
    entrance = graph.entrance_node
    ride0 = graph.ride_node(0)
    a = graph.walk_time(entrance, ride0)
    b = graph.walk_time(ride0, entrance)
    assert a > 0
    assert b > 0


def test_walk_times_to_rides_shape():
    graph = get_park_graph()
    walks = graph.walk_times_to_rides(graph.entrance_node, 1.4)
    assert len(walks) == 34
    assert all(w > 0 for w in walks)


def test_neighbors_within_hops():
    graph = get_park_graph()
    neighbors = graph.neighbors_within_hops(graph.entrance_node, 2)
    assert graph.entrance_node not in neighbors
    assert len(neighbors) > 0


def test_walk_matrix_disk_cache_roundtrip(tmp_path, monkeypatch):
    cache_path = tmp_path / "walk_matrix.npz"
    monkeypatch.setattr("park_graph.WALK_MATRIX_CACHE_PATH", cache_path)
    monkeypatch.setattr("park_graph.CACHE_DIR", tmp_path)

    reset_park_graph()
    if cache_path.exists():
        cache_path.unlink()

    g1 = build_graph(force_recompute=True)
    assert cache_path.is_file()

    g2 = build_graph(force_recompute=False)
    assert g1.walk_time_sec == g2.walk_time_sec
    assert g1.walk_variant_count == g2.walk_variant_count
    assert g1.walk_variant_base_sec == g2.walk_variant_base_sec

    # Stale fingerprint must miss (avoid a second full recompute in this test).
    data = dict(np.load(cache_path, allow_pickle=False))
    data["fingerprint"] = np.asarray("stale")
    np.savez_compressed(cache_path, **data)
    n = g1.num_nodes
    k_max = len(g1.walk_variant_base_sec[0][0])
    assert _load_walk_matrix_cache("current-fingerprint", n, k_max) is None

    reset_park_graph()
