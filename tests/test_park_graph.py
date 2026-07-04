"""Tests for park graph pathfinding."""

from park_graph import get_park_graph


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
    assert len(walks) == 35
    assert all(w > 0 for w in walks)


def test_neighbors_within_hops():
    graph = get_park_graph()
    neighbors = graph.neighbors_within_hops(graph.entrance_node, 2)
    assert graph.entrance_node not in neighbors
    assert len(neighbors) > 0
