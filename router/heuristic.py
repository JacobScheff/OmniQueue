"""Heuristic baseline router with Numba-accelerated batch routing."""

from __future__ import annotations

import numpy as np

import config
from park_types import EXIT_RIDE_ID, ROUTE_IDLE_CODE
from router.numba_routing import route_batch_numba


class HeuristicRouter:
    def route_batch(self, party_ids, parties, rides, graph, now_sec, rng):
        rides.refresh_router_cache()

        if not party_ids:
            return []

        results: list[tuple[int, int | None]] = []
        ids = np.asarray(party_ids, dtype=np.int32)

        for start in range(0, len(ids), config.MAX_ROUTE_BATCH):
            chunk = ids[start : start + config.MAX_ROUTE_BATCH]
            rand_u01 = rng.random(chunk.shape[0], dtype=np.float64)
            targets = route_batch_numba(
                chunk,
                parties.leave_sec,
                parties.location_node_idx,
                parties.effective_speed,
                parties.preference_order,
                parties.balk_sec,
                graph.node_idx_to_ride,
                rides.open_mask,
                rides.wait_arr,
                rides.duration_arr,
                graph.base_walk_to_rides,
                float(config.BASE_WALKING_SPEED),
                int(now_sec),
                rand_u01,
                float(config.IDLE_WALK_PROB),
                int(EXIT_RIDE_ID),
                int(ROUTE_IDLE_CODE),
            )
            for i, pid in enumerate(chunk):
                target = int(targets[i])
                if target == ROUTE_IDLE_CODE:
                    results.append((int(pid), None))
                else:
                    results.append((int(pid), target))

        return results
