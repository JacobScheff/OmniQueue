"""Python entry point for the C++ discrete event simulator."""

from __future__ import annotations

import os

import Park.config as config
from Park.metrics import DayMetrics


def _native_available() -> bool:
    try:
        import _park_sim  # type: ignore[import-not-found]

        return bool(_park_sim.is_available())
    except ImportError:
        return False


def _metrics_from_native(result) -> DayMetrics:
    return DayMetrics(
        total_parties=result.total_parties,
        total_guests=result.total_guests,
        rides_completed=result.rides_completed,
        parties_exited=result.parties_exited,
        breakdown_count=result.breakdown_count,
        wait_variance_samples=list(result.wait_variance_samples),
        mean_wait_samples=list(result.mean_wait_samples),
        wall_time_sec=result.wall_time_sec,
        must_dos_assigned=int(getattr(result, "must_dos_assigned", 0) or 0),
        must_dos_completed=int(getattr(result, "must_dos_completed", 0) or 0),
        preference_score_sum=float(getattr(result, "preference_score_sum", 0.0) or 0.0),
        must_do_latency_sum_sec=float(getattr(result, "must_do_latency_sum_sec", 0.0) or 0.0),
        must_do_latency_count=int(getattr(result, "must_do_latency_count", 0) or 0),
    )


def native_backend_name() -> str:
    return "native" if _native_available() else "unavailable"


def run_day(
    seed: int = 0,
    router: str | None = None,
    backend: str | None = None,
) -> DayMetrics:
    """Run one simulated park day via the C++ extension.

    ``router`` selects which routing strategy to validate ("heuristic" or "ppo").
    The C++ simulator always runs the built-in heuristic router internally; PPO
    routing via the native sim is a Phase 3 deliverable.

    backend:
      - ``auto`` (default): use C++ when built
      - ``native``: require C++ extension
    """
    selected = backend or os.environ.get("OMNIQUEUE_BACKEND", "auto")
    router_name = router or config.ROUTER

    if selected == "python":
        raise RuntimeError(
            "The Python DES was removed. Build the native extension with: pip install -e ."
        )

    if router_name == "ppo":
        raise NotImplementedError(
            "PPO routing is not integrated with the native simulator yet (Phase 3)."
        )
    if router_name not in ("heuristic", None):
        raise ValueError(f"Unknown router: {router_name}")

    if not _native_available():
        raise ImportError(
            "C++ extension _park_sim is not built. Run: pip install -e ."
        )

    import _park_sim  # type: ignore[import-not-found]

    return _metrics_from_native(_park_sim.run_day(seed))


def record_day(seed: int = 0, sample_interval_sec: int = 60):
    """Simulate one park day and return a ``_park_sim.DayRecording`` for visualization."""
    if not _native_available():
        raise ImportError(
            "C++ extension _park_sim is not built. Run: pip install -e ."
        )

    import _park_sim  # type: ignore[import-not-found]

    return _park_sim.record_day(seed, sample_interval_sec)
