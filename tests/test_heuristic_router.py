"""Tests for heuristic router."""

import numpy as np

import config
from park_graph import get_park_graph
from parties import PartyPool
from rides import RideManager
from router.heuristic import HeuristicRouter


def test_router_returns_ride_or_exit():
    graph = get_park_graph()
    rides = RideManager(__import__("random").Random(0))
    rides.refresh_router_cache()

    pool = PartyPool(graph)
    schedules = pool.spawn_day(np.random.default_rng(0))
    assert schedules
    party_id = schedules[0][1]

    router = HeuristicRouter()
    results = router.route_batch([party_id], pool, rides, graph, now_sec=3600, rng=np.random.default_rng(0))
    assert len(results) == 1
    pid, target = results[0]
    assert pid == party_id
    assert target is None or target >= 0 or target == -1
