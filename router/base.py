"""Router protocol and factory (Phase 3 PPO)."""

from __future__ import annotations

from typing import Protocol

import config


class Router(Protocol):
    def route_batch(
        self,
        party_ids: list[int],
        parties,
        rides,
        graph,
        now_sec: int,
        rng,
    ) -> list[tuple[int, int | None]]:
        """Return list of (party_id, target_ride_id | None for idle walk | EXIT)."""
        ...


def get_router(name: str | None = None) -> Router:
    router_name = name or config.ROUTER
    if router_name == "heuristic":
        raise RuntimeError(
            "Heuristic routing runs inside the C++ simulator. Call simulator.run_day() instead."
        )
    if router_name == "ppo":
        from router.ppo import PPORouter

        return PPORouter()
    raise ValueError(f"Unknown router: {router_name}")
