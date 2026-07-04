"""Heuristic baseline router with preference-ordered balking."""

from __future__ import annotations

import config
from park_types import EXIT_RIDE_ID


class HeuristicRouter:
    def route_batch(self, party_ids, parties, rides, graph, now_sec, rng):
        rides.refresh_router_cache()
        open_mask = rides.open_mask
        wait_times = rides.wait_arr
        durations = rides.duration_arr

        results = []
        for party_id in party_ids:
            party = parties.get(party_id)
            if parties.should_leave(party, now_sec):
                results.append((party_id, EXIT_RIDE_ID))
                continue

            remaining = parties.time_remaining(party, now_sec)
            current_ride = graph.node_to_ride(party.location_node)
            walks = graph.walk_times_to_rides(party.location_node, party.effective_speed)

            chosen = self._pick_by_balking(
                party, open_mask, wait_times, durations, walks, remaining, current_ride
            )
            if chosen is not None:
                results.append((party_id, chosen))
                continue

            if rng.random() < config.IDLE_WALK_PROB:
                results.append((party_id, None))
                continue

            forced = self._force_pick(
                party, open_mask, wait_times, durations, walks, remaining, current_ride
            )
            results.append((party_id, forced))

        return results

    def _pick_by_balking(self, party, open_mask, wait_times, durations, walks, remaining, current_ride):
        for ride_id in party.preference_order:
            if current_ride is not None and ride_id == current_ride:
                continue
            if not open_mask[ride_id]:
                continue
            walk = int(walks[ride_id])
            if walk + wait_times[ride_id] + durations[ride_id] > remaining:
                continue
            if wait_times[ride_id] <= party.balk_sec[ride_id]:
                return ride_id
        return None

    def _force_pick(self, party, open_mask, wait_times, durations, walks, remaining, current_ride):
        for ride_id in party.preference_order:
            if current_ride is not None and ride_id == current_ride:
                continue
            if not open_mask[ride_id]:
                continue
            walk = int(walks[ride_id])
            if walk + wait_times[ride_id] + durations[ride_id] > remaining:
                continue
            return ride_id
        return EXIT_RIDE_ID
