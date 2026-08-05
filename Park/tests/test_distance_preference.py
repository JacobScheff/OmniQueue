"""Tests for party distance_preference (walk tolerance)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from Park import config
from Park.companion.server.obs import CompanionState, build_live_observation, default_preference_weights
from Park.companion.server.recommend import build_action_mask_numpy
from Park.companion.server.waits import LiveBoard, RideLiveStatus
from Park.training.features import (
    FLAT_OBS_DIM,
    GUEST_FEAT_AT_RIDE_NODE,
    GUEST_FEAT_DIM,
    GUEST_FEAT_DISTANCE_PREF,
    GUEST_FEAT_TIME_LEFT,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    RIDE_FEAT_OPEN,
    RIDE_FEAT_WALK,
    build_action_mask,
    distance_pref_walk_inflate,
    inflate_walk_feat,
)
from Park.training.route_reward import planned_walk_penalty, realized_walk_penalty, pad_route
from Park.simulator import native_backend_name


def _board_all_open(wait_min: float = 15.0) -> LiveBoard:
    rides = [
        RideLiveStatus(
            ride_id=i,
            name=r["name"],
            wait_min=wait_min,
            status="OPERATING",
            open=True,
        )
        for i, r in enumerate(config.RIDES)
    ]
    return LiveBoard(fetched_at=0.0, rides=rides)


def test_flat_obs_dim_and_guest_distance_slot():
    assert GUEST_FEAT_DIM == 44
    assert GUEST_FEAT_DISTANCE_PREF == 43
    assert FLAT_OBS_DIM == 44 + 34 * 11 + 3


def test_walk_inflate_monotonic_in_one_minus_d():
    assert distance_pref_walk_inflate(1.0) == pytest.approx(1.0)
    assert distance_pref_walk_inflate(0.0) > distance_pref_walk_inflate(0.5)
    assert distance_pref_walk_inflate(0.5) > distance_pref_walk_inflate(1.0)
    low = inflate_walk_feat(0.2, 0.0)
    high = inflate_walk_feat(0.2, 1.0)
    assert low > high
    assert high == pytest.approx(0.2)


def test_walk_shaping_scales_with_distance_pref():
    route = pad_route([2, 3, 9, 10, 11])
    full = planned_walk_penalty(route, distance_pref=0.0)
    half = planned_walk_penalty(route, distance_pref=0.5)
    none = planned_walk_penalty(route, distance_pref=1.0)
    assert full > 0.0
    assert half == pytest.approx(0.5 * full, rel=1e-5)
    assert none == pytest.approx(0.0)
    assert realized_walk_penalty(600.0, 0.0) > realized_walk_penalty(600.0, 0.5)
    assert realized_walk_penalty(600.0, 1.0) == pytest.approx(0.0)


def test_training_mask_still_blocks_already_here():
    guest = torch.zeros(1, GUEST_FEAT_DIM)
    ride = torch.zeros(1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = torch.zeros(1, 3)
    guest[..., GUEST_FEAT_TIME_LEFT] = 0.5
    guest[..., GUEST_FEAT_AT_RIDE_NODE] = 1.0
    guest[..., GUEST_FEAT_DISTANCE_PREF] = 0.5
    ride[..., RIDE_FEAT_OPEN] = 1.0
    ride[..., 0, RIDE_FEAT_WALK] = 0.0  # standing at ride 0
    ride[..., 1, RIDE_FEAT_WALK] = 0.1
    mask = build_action_mask(guest, ride, env)
    assert not bool(mask[0, 0])
    assert bool(mask[0, 1])


def test_companion_mask_allows_current_ride():
    guest = np.zeros((1, GUEST_FEAT_DIM), dtype=np.float32)
    ride = np.zeros((1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM), dtype=np.float32)
    env = np.zeros((1, 3), dtype=np.float32)
    guest[..., GUEST_FEAT_TIME_LEFT] = 0.5
    guest[..., GUEST_FEAT_AT_RIDE_NODE] = 1.0
    guest[..., GUEST_FEAT_DISTANCE_PREF] = 0.5
    ride[..., RIDE_FEAT_OPEN] = 1.0
    ride[..., 0, RIDE_FEAT_WALK] = 0.0
    ride[..., 1, RIDE_FEAT_WALK] = 0.1
    mask = build_action_mask_numpy(guest, ride, env)
    assert bool(mask[0, 0])
    assert bool(mask[0, 1])


def test_companion_obs_packs_distance_and_inflates_walk():
    weights = default_preference_weights()
    state_low = CompanionState(
        preference_weights=weights,
        must_dos=np.zeros(NUM_RIDES, dtype=np.uint8),
        history=np.zeros(NUM_RIDES, dtype=np.int32),
        location_node_id=config.NODE_ENTRANCE,
        distance_preference=0.0,
    )
    state_high = CompanionState(
        preference_weights=weights,
        must_dos=np.zeros(NUM_RIDES, dtype=np.uint8),
        history=np.zeros(NUM_RIDES, dtype=np.int32),
        location_node_id=config.NODE_ENTRANCE,
        distance_preference=1.0,
    )
    flat_low, meta_low = build_live_observation(state_low, _board_all_open(), now_sec=3600)
    flat_high, meta_high = build_live_observation(state_high, _board_all_open(), now_sec=3600)
    assert flat_low.shape == (FLAT_OBS_DIM,)
    assert meta_low["distance_preference"] == pytest.approx(0.0)
    assert meta_high["distance_preference"] == pytest.approx(1.0)
    assert flat_low[GUEST_FEAT_DISTANCE_PREF] == pytest.approx(0.0)
    assert flat_high[GUEST_FEAT_DISTANCE_PREF] == pytest.approx(1.0)
    ride_low = flat_low[
        GUEST_FEAT_DIM : GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    ].reshape(NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    ride_high = flat_high[
        GUEST_FEAT_DIM : GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    ].reshape(NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    # Far rides should score as longer walks when d is low.
    assert float(ride_low[:, RIDE_FEAT_WALK].mean()) > float(ride_high[:, RIDE_FEAT_WALK].mean())


def test_companion_at_ride_keeps_ride_legal():
    ride_id = 12
    weights = default_preference_weights()
    state = CompanionState(
        preference_weights=weights,
        must_dos=np.zeros(NUM_RIDES, dtype=np.uint8),
        history=np.zeros(NUM_RIDES, dtype=np.int32),
        location_node_id=config.ride_node_id(ride_id),
        distance_preference=0.5,
    )
    flat, meta = build_live_observation(state, _board_all_open(), now_sec=3600)
    assert meta["at_ride_id"] == ride_id
    guest = flat[:GUEST_FEAT_DIM].reshape(1, GUEST_FEAT_DIM)
    ride = flat[
        GUEST_FEAT_DIM : GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    ].reshape(1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = flat[-3:].reshape(1, 3)
    mask = build_action_mask_numpy(guest, ride, env)
    assert bool(mask[0, ride_id])


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_heuristic_distance_balk_rejects_far_when_d_low():
    import _park_sim
    from Park.park_graph import get_park_graph

    n = _park_sim.NUM_RIDES
    graph = get_park_graph()
    walks = graph.walk_times_to_rides(config.NODE_ENTRANCE, config.BASE_WALKING_SPEED)
    near = int(np.argmin(walks))
    far = int(np.argmax(walks))
    assert walks[far] > config.DISTANCE_PREF_NEAR_WALK_SEC

    # Preference order: far ride first, then near — Pass 1 would pick far without distance balk.
    order = [far, near] + [i for i in range(n) if i not in (far, near)]
    prefs = [0.01] * n
    prefs[far] = 0.5
    prefs[near] = 0.2
    s = sum(prefs)
    prefs = [p / s for p in prefs]
    balk = [40 * 60.0] * n
    args = {
        "now_sec": 0,
        "leave_sec": 54_000,
        "node_idx": 0,
        "speed": 1.4,
        "preference_order": order,
        "preferences": prefs,
        "balk_sec": balk,
        "ride_history": [0] * n,
        "open_mask": [1] * n,
        "wait_times": [5 * 60.0] * n,
        "durations": [120] * n,
        "rand_u01": 1.0,
    }
    chosen_high = _park_sim.route_one_for_test(**args, distance_preference=1.0)
    assert chosen_high == far
    chosen_low = _park_sim.route_one_for_test(**args, distance_preference=0.0)
    assert chosen_low == near


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_native_obs_includes_distance_preference():
    import _park_sim
    from Park.training.features import GUEST_FEAT_DISTANCE_PREF, GUEST_FEAT_DIM

    env = _park_sim.ParkEnv(0)
    obs = env.reset(0)
    guest = list(obs.guest)
    assert len(guest) == GUEST_FEAT_DIM
    assert 0.0 <= guest[GUEST_FEAT_DISTANCE_PREF] <= 1.0


def test_guest_proj_widen_warm_start():
    from Park.training.checkpoint import TrainConfig, _widen_guest_proj_state, default_model

    cfg = TrainConfig()
    # Build a "legacy" guest_proj input = (GUEST-1)+ENV
    old_in = (GUEST_FEAT_DIM - 1) + cfg.environment_dynamic_feat_dim
    new_model = default_model("cpu")
    fake_old = {
        "guest_proj.0.weight": torch.randn(
            new_model.guest_proj[0].out_features, old_in
        ),
        "guest_proj.0.bias": torch.randn(new_model.guest_proj[0].out_features),
    }
    # Seed known values in last env column and first guest column.
    fake_old["guest_proj.0.weight"][:, 0] = 3.0
    fake_old["guest_proj.0.weight"][:, -1] = 7.0
    widened, notes = _widen_guest_proj_state(new_model, fake_old)
    assert any("widened_guest_proj" in n for n in notes)
    w = widened["guest_proj.0.weight"]
    assert w.shape[1] == GUEST_FEAT_DIM + cfg.environment_dynamic_feat_dim
    assert torch.allclose(w[:, 0], torch.full_like(w[:, 0], 3.0))
    assert torch.allclose(w[:, -1], torch.full_like(w[:, -1], 7.0))
    # New guest column (index GUEST_FEAT_DIM-1) is zero.
    assert torch.allclose(w[:, GUEST_FEAT_DIM - 1], torch.zeros_like(w[:, 0]))
