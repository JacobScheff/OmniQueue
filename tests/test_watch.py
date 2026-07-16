"""Tests for watch mode: mid-day prefs, timeline helpers, PPO probs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import config
from play.driver import make_focal_config
from play.session import FocalProfile
from simulator import native_backend_name
from watch.session import DecisionMark
from watch.timeline import (
    TimelineState,
    completion_counts_at,
    mark_index_for_click,
    scrub_to_frac,
)


pytestmark_native = pytest.mark.skipif(
    native_backend_name() != "native", reason="C++ extension not built"
)


def _profile() -> FocalProfile:
    w = np.ones(config.NUM_RIDES, dtype=np.float32)
    w[0] = 10.0
    w[6] = 8.0
    md = np.zeros(config.NUM_RIDES, dtype=np.uint8)
    md[0] = 1
    return FocalProfile(
        spawn_sec=0,
        leave_sec=config.DAY_SECONDS,
        preference_weights=w,
        must_dos=md,
    )


def test_require_watch_apis_message():
    from watch.driver import _require_watch_apis

    class FakeEnv:
        pass

    class FakeMod:
        ParkEnv = FakeEnv

    with pytest.raises(RuntimeError, match="pip install -e"):
        _require_watch_apis(FakeMod())


@pytestmark_native
def test_play_update_focal_preferences_keeps_history():
    import _park_sim

    env = _park_sim.ParkEnv(7)
    profile = _profile()
    env.reset_play(
        7,
        make_focal_config(profile),
        crowd_auto_heuristic=True,
        focal_policy=1,  # heuristic focal
        soft_human_leave=False,
        enable_recording=False,
        sample_interval_sec=60,
    )
    # Advance until some progress / possible completion
    for _ in range(500):
        step = env.play_advance()
        if step.done:
            break
        # heuristic crowd + heuristic focal should never need PPO/human
        assert not step.needs_ppo_batch
        assert not step.needs_human

    hist_before = np.asarray(env.play_focal_ride_history(), dtype=np.int16).copy()
    stats_before = env.play_focal_stats()
    now_before = env.play_now_sec()

    new_w = np.ones(config.NUM_RIDES, dtype=np.float32)
    new_w[3] = 50.0
    new_md = np.zeros(config.NUM_RIDES, dtype=np.uint8)
    new_md[3] = 1
    profile.preference_weights = new_w
    profile.must_dos = new_md
    env.play_update_focal_preferences(make_focal_config(profile))

    hist_after = np.asarray(env.play_focal_ride_history(), dtype=np.int16)
    stats_after = env.play_focal_stats()
    assert env.play_now_sec() == now_before
    assert np.array_equal(hist_before, hist_after)
    assert int(stats_after.rides_completed) == int(stats_before.rides_completed)
    assert int(stats_after.must_dos_assigned) == 1
    # Preference mass should concentrate on ride 3 after normalize+boost
    prefs = np.asarray(stats_after.preferences, dtype=np.float32)
    assert float(prefs[3]) == max(float(x) for x in prefs)


@pytestmark_native
def test_play_focal_state_is_valid():
    import _park_sim

    env = _park_sim.ParkEnv(3)
    env.reset_play(
        3,
        make_focal_config(_profile()),
        crowd_auto_heuristic=True,
        focal_policy=1,
        soft_human_leave=False,
        enable_recording=False,
    )
    state = int(env.play_focal_state())
    assert state in (0, 1, 2, 4, 8, 16)


def test_timeline_pref_edit_gate():
    tl = TimelineState(playhead_sec=100.0, frontier_sec=100.0, paused=True)
    assert tl.can_edit_prefs()
    tl.playhead_sec = 50.0
    assert not tl.can_edit_prefs()
    tl.playhead_sec = 100.0
    tl.paused = False
    assert not tl.can_edit_prefs()


def test_completion_counts_and_scrub():
    class Ev:
        def __init__(self, party_id, sec, ride_id):
            self.party_id = party_id
            self.sec = sec
            self.ride_id = ride_id

    comps = [Ev(0, 10, 1), Ev(0, 50, 1), Ev(0, 80, 2), Ev(1, 20, 1)]
    assert completion_counts_at(comps, 0, 40, 5) == [0, 1, 0, 0, 0]
    assert completion_counts_at(comps, 0, 100, 5) == [0, 2, 1, 0, 0]
    assert scrub_to_frac(0.5, 1000, 400) == 400.0
    assert scrub_to_frac(0.1, 1000, 400) == 100.0


def test_mark_click_hit():
    marks = [
        (0, DecisionMark(sec=100, scope="focal", party_id=0, action=1, probs=np.zeros(36))),
        (1, DecisionMark(sec=500, scope="crowd", party_id=2, action=2, probs=np.zeros(36))),
    ]
    # slider x=0,w=1000, day=1000 → mark at sec 100 is x=100
    assert mark_index_for_click(marks, 100, 0, 1000, 1000.0, hit_px=5) == 0
    assert mark_index_for_click(marks, 500, 0, 1000, 1000.0, hit_px=5) == 1
    assert mark_index_for_click(marks, 300, 0, 1000, 1000.0, hit_px=5) is None


def test_ppo_act_with_probs(tmp_path: Path):
    pytest.importorskip("torch")
    from router.ppo import PPOPolicy
    from training.checkpoint import default_model, save_checkpoint
    from training.features import FLAT_OBS_DIM, GUEST_FEAT_DIM, NUM_ACTIONS, RIDE_DYNAMIC_FEAT_DIM

    ckpt = tmp_path / "tiny.pt"
    save_checkpoint(ckpt, default_model(), None, step=1, extra={"phase": "test"})
    policy = PPOPolicy(ckpt, device="cpu")
    obs = np.zeros(FLAT_OBS_DIM, dtype=np.float32)
    # Make rides look open / feasible enough for a non-degenerate mask.
    ride = obs[
        GUEST_FEAT_DIM : GUEST_FEAT_DIM + config.NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    ].reshape(config.NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    ride[:, 0] = 0.01  # wait fraction
    ride[:, 2] = 1.0  # open
    ride[:, 3] = 0.05  # duration
    ride[:, 5] = 0.01  # walk
    obs[37] = 0.9  # time left
    action, probs = policy.act_with_probs(obs)
    assert 0 <= action < NUM_ACTIONS
    assert probs.shape == (NUM_ACTIONS,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert int(probs.argmax()) == action


@pytestmark_native
def test_watch_driver_records_focal_decisions(tmp_path: Path):
    pytest.importorskip("torch")
    from training.checkpoint import default_model, save_checkpoint
    from watch.driver import WatchDriver

    ckpt = tmp_path / "watch.pt"
    save_checkpoint(ckpt, default_model(), None, step=1, extra={"phase": "test"})
    driver = WatchDriver(
        seed=11,
        profile=_profile(),
        crowd_router="heuristic",
        checkpoint=ckpt,
        device="cpu",
        sample_interval_sec=120,
    )
    result = driver.advance_until(
        stop_on_queue=False,
        stop_on_focal_decision=True,
        max_batches=2000,
        min_time_advance=0,
    )
    assert not result.done
    focal = [d for d in driver.decisions if d.scope == "focal"]
    assert len(focal) >= 1
    assert focal[0].probs.shape[0] == 36
    assert result.focal_decisions
    assert driver.recording() is not None
