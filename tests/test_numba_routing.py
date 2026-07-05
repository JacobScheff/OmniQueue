"""Parity tests for Numba routing kernel."""

import numpy as np

import config
from park_graph import get_park_graph
from parties import PartyPool
from rides import RideManager
from router.numba_routing import route_batch_numba
from park_types import EXIT_RIDE_ID, ROUTE_IDLE_CODE


def test_numba_route_batch_smoke():
    graph = get_park_graph()
    rides = RideManager(__import__("random").Random(1))
    rides.refresh_router_cache()

    pool = PartyPool(graph)
    pool.spawn_day(np.random.default_rng(1))
    party_ids = np.array([0, 1, 2], dtype=np.int32)
    rand_u01 = np.array([0.1, 0.9, 0.5], dtype=np.float64)

    targets = route_batch_numba(
        party_ids,
        pool.leave_sec,
        pool.location_node_idx,
        pool.effective_speed,
        pool.preference_order,
        pool.balk_sec,
        graph.node_idx_to_ride,
        rides.open_mask,
        rides.wait_arr,
        rides.duration_arr,
        graph.base_walk_to_rides,
        float(config.BASE_WALKING_SPEED),
        3600,
        rand_u01,
        float(config.IDLE_WALK_PROB),
        int(EXIT_RIDE_ID),
        int(ROUTE_IDLE_CODE),
    )

    assert targets.shape == (3,)
    assert all(t >= EXIT_RIDE_ID for t in targets)
