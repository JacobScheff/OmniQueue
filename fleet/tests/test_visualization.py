"""Tests for day recording used by the fleet visualizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from fleet.simulator import native_backend_name, record_day, record_day_ppo


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_smoke():
    rec = record_day(
        seed=11,
        sample_interval_sec=300,
        horizon_sec=3600,
        num_requests=200,
        num_vehicles=25,
        num_intersections=80,
    )
    assert len(rec.city.nodes) > 0
    assert len(rec.city.edges) > 0
    assert len(rec.requests) == 200
    assert len(rec.trips) > 0
    assert len(rec.samples) > 0
    assert rec.metrics.requests_completed > 0
    assert rec.horizon_sec == 3600


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_trip_fields():
    rec = record_day(
        seed=3,
        sample_interval_sec=600,
        horizon_sec=3600,
        num_requests=100,
        num_vehicles=15,
    )
    t = rec.trips[0]
    assert t.end_sec >= t.start_sec
    n = len(rec.city.nodes)
    assert 0 <= int(t.from_node) < n
    assert 0 <= int(t.to_node) < n
    assert 0 <= int(t.vehicle_id) < rec.num_vehicles


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_completions_match_metrics():
    rec = record_day(
        seed=9,
        sample_interval_sec=600,
        horizon_sec=3600,
        num_requests=150,
        num_vehicles=20,
    )
    completed = sum(1 for r in rec.requests if int(r.status) == 4)
    assert completed == rec.metrics.requests_completed


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_replay_helpers():
    from fleet.visualize import (
        ReplayState,
        active_trips_at,
        trip_position,
        vehicle_state_at,
    )

    rec = record_day(
        seed=7,
        sample_interval_sec=300,
        horizon_sec=3600,
        num_requests=120,
        num_vehicles=20,
    )
    state = ReplayState.from_recording(rec)
    assert len(state.sorted_vehicle_ids) == rec.num_vehicles

    mid = 1800.0
    trips = active_trips_at(state, mid)
    for t in trips[:5]:
        x, y = trip_position(state, t, mid)
        assert 0 <= x <= state.city_width
        assert 0 <= y <= state.city_height

    g = vehicle_state_at(state, 0, mid)
    assert g is not None
    assert "status" in g
    assert "pos" in g


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_ppo_smoke(tmp_path: Path):
    import torch

    from fleet.model import VehicleRouter

    ckpt_path = tmp_path / "ppo_smoke.pt"
    model = VehicleRouter(use_graph_encoder=False)
    torch.save(
        {
            "model": model.state_dict(),
            "update": 0,
            "config": {
                "num_envs": 1,
                "num_vehicles": 10,
                "num_requests": 40,
                "num_intersections": 40,
                "horizon_sec": 900,
            },
        },
        ckpt_path,
    )

    rec = record_day_ppo(
        seed=5,
        checkpoint=ckpt_path,
        sample_interval_sec=120,
        device="cpu",
    )
    assert len(rec.city.nodes) > 0
    assert len(rec.requests) == 40
    assert len(rec.trips) > 0
    assert len(rec.samples) > 0
    assert rec.horizon_sec == 900
    assert rec.num_vehicles == 10


def test_visualize_window_is_scaled_down():
    from fleet.visualize import (
        CONTROL_HEIGHT,
        MAP_HEIGHT,
        MAP_LOGICAL,
        MAP_WIDTH,
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
        SIDEBAR_WIDTH,
        UI_SCALE,
        _s,
        _xy,
    )

    assert UI_SCALE == pytest.approx(0.85)
    assert MAP_WIDTH == int(MAP_LOGICAL * UI_SCALE)
    assert MAP_HEIGHT == int(MAP_LOGICAL * UI_SCALE)
    assert SIDEBAR_WIDTH == int(320 * UI_SCALE)
    assert CONTROL_HEIGHT == int(70 * UI_SCALE)
    assert SCREEN_WIDTH == MAP_WIDTH + SIDEBAR_WIDTH
    assert SCREEN_HEIGHT == MAP_HEIGHT + CONTROL_HEIGHT
    assert _xy(1000, 1000) == (MAP_WIDTH, MAP_HEIGHT)
    assert _s(100) == 85
