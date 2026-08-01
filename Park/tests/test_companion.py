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
    assert len(out["distributions_by_slot"]) >= 1
    assert out["distributions_by_slot"][0] == out["distribution"]
    assert out["forced_first"] is None
    assert out["natural_recommended"]["action_id"] == out["recommended"]["action_id"]
    assert out["model"]["supports_force_first"] is True
    assert out["model"]["supports_slot_distributions"] is True

    natural0 = out["recommended"]["action_id"]
    force_id = 0 if natural0 != 0 else 1
    # Skip if that ride is illegal under the stub mask.
    legal_ids = [r["action_id"] for r in out["distribution"] if r["legal"] and r["is_ride"]]
    if force_id not in legal_ids and legal_ids:
        force_id = legal_ids[0] if legal_ids[0] != natural0 else legal_ids[-1]
    forced = rec.recommend(flat, force_first=force_id)
    assert forced["forced_first"] == force_id
    assert forced["route"][0]["action_id"] == force_id
    assert forced["recommended"]["action_id"] == force_id
    assert forced["natural_recommended"]["action_id"] == natural0
    assert len(forced["distributions_by_slot"]) == len(forced["route"])


def test_model_registry_versions(tmp_path, monkeypatch):
    from Park.companion import settings
    from Park.companion.server.recommend import ModelRegistry, Recommender

    v1 = tmp_path / "v1.pt"
    v2 = tmp_path / "v2.pt"
    Recommender(checkpoint=v1, device="cpu")
    Recommender(checkpoint=v2, device="cpu")
    monkeypatch.setattr(settings, "MODELS", {"v1": v1, "v2": v2})
    monkeypatch.setattr(settings, "DEFAULT_MODEL_VERSION", "v2")
    reg = ModelRegistry(device="cpu")
    assert [m["id"] for m in reg.versions()] == ["v1", "v2"]
    assert reg.get(None).version == "v2"
    assert reg.get("v1").checkpoint_path == v1
