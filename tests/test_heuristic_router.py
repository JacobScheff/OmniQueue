"""Tests for heuristic router."""

import numpy as np

import config
from park_graph import get_park_graph
from park_types import Party, PartyState
from parties import PartyPool, _compute_balk_sec, _compute_preference_order
from rides import RideManager
from router.heuristic import HeuristicRouter


def _make_party(party_id=0):
    prefs = [1.0 / config.NUM_RIDES] * config.NUM_RIDES
    must_do = [False] * config.NUM_RIDES
    return Party(
        party_id=party_id,
        party_size=3,
        spawn_sec=0,
        leave_sec=config.DAY_SECONDS,
        effective_speed=1.4,
        preferences=prefs,
        must_do=must_do,
        must_do_remaining=list(must_do),
        ride_history=[0] * config.NUM_RIDES,
        preference_order=_compute_preference_order(prefs, must_do),
        balk_sec=_compute_balk_sec(prefs),
        location_node=get_park_graph().entrance_node,
        state=PartyState.WALKING,
    )


def test_router_returns_ride_or_exit():
    graph = get_park_graph()
    rides = RideManager(__import__("random").Random(0))
    rides.refresh_router_cache()

    pool = PartyPool(graph)
    pool.parties = [_make_party()]

    router = HeuristicRouter()
    results = router.route_batch([0], pool, rides, graph, now_sec=3600, rng=np.random.default_rng(0))
    assert len(results) == 1
    party_id, target = results[0]
    assert party_id == 0
    assert target is None or target >= 0 or target == -1
