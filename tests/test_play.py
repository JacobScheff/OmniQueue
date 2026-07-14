"""Tests for interactive play hybrid APIs and scoring."""

from __future__ import annotations

import numpy as np
import pytest

from play.driver import HybridDriver, run_heuristic_focal_day
from play.scoring import normalize_weights
from play.session import FocalProfile
from simulator import native_backend_name


pytestmark = pytest.mark.skipif(
    native_backend_name() != "native", reason="C++ extension not built"
)


def _profile() -> FocalProfile:
    w = np.ones(34, dtype=np.float32)
    w[0] = 10.0
    w[6] = 8.0
    md = np.zeros(34, dtype=np.uint8)
    md[0] = 1
    return FocalProfile(spawn_sec=300, leave_sec=36000, preference_weights=w, must_dos=md)


def test_normalize_weights_must_do_boost():
    w = np.ones(34, dtype=np.float32)
    md = np.zeros(34, dtype=np.uint8)
    md[3] = 1
    out = normalize_weights(w, md)
    assert out.shape == (34,)
    assert abs(float(out.sum()) - 1.0) < 1e-5
    assert out[3] > out[0]


def test_run_heuristic_focal_day_pref_metrics():
    run = run_heuristic_focal_day(seed=3, profile=_profile(), record=False)
    assert run.park.rides_completed > 0
    assert run.focal.rides_completed > 0
    assert run.focal.must_dos_assigned == 1
    assert run.focal.preference_score > 0.0
    assert run.settings.crowd_router == "heuristic"
    assert run.settings.focal_router == "heuristic"


def test_hybrid_human_can_exit_immediately():
    driver = HybridDriver(
        seed=5,
        profile=_profile(),
        crowd_router="heuristic",
        focal_router="human",
        enable_recording=False,
        soft_human_leave=True,
    )
    decision = driver.advance()
    assert decision is not None
    assert decision.now_sec == 300
    driver.apply_human_action(34)  # exit
    assert driver.advance() is None
    assert driver.done
    run = driver.to_session_run()
    assert run.focal.exited
    assert run.focal.rides_completed == 0


def test_hybrid_heuristic_focal_completes():
    driver = HybridDriver(
        seed=9,
        profile=_profile(),
        crowd_router="heuristic",
        focal_router="heuristic",
        enable_recording=False,
        soft_human_leave=False,
    )
    run = driver.run_headless()
    assert run.focal.rides_completed > 0
    assert run.park.rides_completed > 0


def test_focal_uses_exact_enter_time():
    import _park_sim

    from play.driver import make_focal_config

    profile = _profile()
    profile.spawn_sec = 12_345
    profile.leave_sec = 40_000
    result = _park_sim.run_play_day(
        2, make_focal_config(profile), sample_interval_sec=60, record=False
    )
    assert int(result.focal.spawn_sec) == 12_345
    assert int(result.focal.leave_sec) == 40_000


def test_run_ai_compare_cell_heuristic():
    from play.benchmark import run_ai_compare_cell
    from play.session import SessionStore

    store = SessionStore()
    run = run_ai_compare_cell(
        seed=4,
        profile=_profile(),
        crowd_router="heuristic",
        focal_router="heuristic",
        label="H-crowd / H-guest",
        checkpoint=None,
        store=store,
    )
    assert run.settings.label == "H-crowd / H-guest"
    assert run.focal.rides_completed > 0
    assert len(store.runs) == 1
