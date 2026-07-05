"""PPO router stub for Phase 3."""

from __future__ import annotations


class PPORouter:
    def route_batch(self, party_ids, parties, rides, graph, now_sec, rng):
        raise NotImplementedError("PPO router is implemented in Phase 3. Set config.ROUTER='heuristic' for Phase 1.")
