"""Unit tests for live companion observation + ride mapping."""

from __future__ import annotations

import numpy as np
import pytest

from Park.companion.server.obs import (
    CompanionState,
    build_live_observation,
    default_preference_weights,
    normalize_preferences,
    resolve_location_node_id,
)
from Park.companion.server.ride_map import (
    build_config_name_index,
    resolve_ride_index,
)
from Park.companion.server.waits import LiveBoard, RideLiveStatus
from Park import config
from Park.training.features import FLAT_OBS_DIM, NUM_ACTIONS, NUM_RIDES


def _board_all_open(wait_min: float = 20.0) -> LiveBoard:
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


def test_normalize_preferences_sums_to_one():
    w = default_preference_weights()
    prefs = normalize_preferences(w)
    assert prefs.shape == (NUM_RIDES,)
    assert prefs.sum() == pytest.approx(1.0, abs=1e-5)


def test_resolve_location_keys():
    assert resolve_location_node_id("entrance") == config.NODE_ENTRANCE
    assert resolve_location_node_id("hub:7") == 7
    assert resolve_location_node_id("ride:0") == config.ride_node_id(0)


def test_ride_name_aliases():
    index = build_config_name_index()
    assert resolve_ride_index(None, "Indiana Jones™ Adventure", index) == 7
    assert resolve_ride_index(None, "Mickey & Minnie's Runaway Railway", index) == 25
    assert resolve_ride_index(None, "Star Tours - The Adventures Continue", index) == 29


def test_build_live_observation_shape_and_mask_inputs():
    weights = default_preference_weights()
    must = np.zeros(NUM_RIDES, dtype=np.uint8)
    must[0] = 1
    history = np.zeros(NUM_RIDES, dtype=np.int32)
    history[5] = 2
    state = CompanionState(
        preference_weights=weights,
        must_dos=must,
        history=history,
        location_node_id=config.NODE_FANTASY_HUB,
        leave_sec=config.DAY_SECONDS,
        spawn_sec=0,
        party_size=2,
    )
    flat, meta = build_live_observation(state, _board_all_open(), now_sec=3 * 3600)
    assert flat.shape == (FLAT_OBS_DIM,)
    assert np.isfinite(flat).all()
    assert meta["must_remaining"][0] == 1
    assert meta["must_remaining"][5] == 0
    # history ride feat
    ride = flat[46 : 46 + NUM_RIDES * 8].reshape(NUM_RIDES, 8)
    assert ride[5, 6] == pytest.approx(0.2)
    assert ride[0, 7] == pytest.approx(1.0)


def test_recommender_stub_smoke(tmp_path):
    from Park.companion.server.recommend import Recommender

    path = tmp_path / "stub.pt"
    rec = Recommender(checkpoint=path, device="cpu")
    assert path.is_file()
    weights = default_preference_weights()
    state = CompanionState(
        preference_weights=weights,
        must_dos=np.zeros(NUM_RIDES, dtype=np.uint8),
        history=np.zeros(NUM_RIDES, dtype=np.int32),
        location_node_id=config.NODE_ENTRANCE,
    )
    flat, _ = build_live_observation(state, _board_all_open(), now_sec=3600)
    out = rec.recommend(flat)
    assert out["recommended"]["action_id"] in range(NUM_ACTIONS)
    assert len(out["distribution"]) == NUM_ACTIONS
    assert abs(sum(r["prob"] for r in out["distribution"]) - 1.0) < 1e-3
