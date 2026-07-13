"""Tests for day recording used by the visualizer."""

from __future__ import annotations

import pytest

from simulator import native_backend_name, record_day


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_smoke():
    rec = record_day(seed=11, sample_interval_sec=300)
    assert len(rec.parties) > 0
    assert len(rec.walks) > 0
    assert len(rec.ride_samples) > 0
    assert rec.metrics.rides_completed > 0
    assert rec.metrics.total_parties == len(rec.parties)


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_walk_fields():
    rec = record_day(seed=3, sample_interval_sec=600)
    w = rec.walks[0]
    assert w.end_sec >= w.start_sec
    assert w.planned_end_sec >= w.start_sec
    assert 0 <= int(w.from_idx) < 47
    assert 0 <= int(w.to_idx) < 47
    assert 0 <= int(getattr(w, "path_variant", 0)) < 8


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_uses_multiple_path_variants():
    rec = record_day(seed=42, sample_interval_sec=600)
    variants = {int(w.path_variant) for w in rec.walks}
    # With randomization enabled, some walks should use non-zero variants.
    assert max(variants) >= 0
    assert any(int(w.path_variant) > 0 for w in rec.walks)


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_ride_samples_shape():
    rec = record_day(seed=5, sample_interval_sec=900)
    sample = rec.ride_samples[0]
    assert len(sample.wait) == 34
    assert len(sample.broken) == 34
    assert len(sample.queue_len) == 34
    assert sample.sec >= 0


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_record_day_completions_match_metrics():
    rec = record_day(seed=9, sample_interval_sec=600)
    assert len(rec.ride_completions) == rec.metrics.rides_completed


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_replay_helpers():
    from visualize import ReplayState, active_walks_at, build_node_coords, party_state_at, walk_position

    rec = record_day(seed=7, sample_interval_sec=300)
    state = ReplayState.from_recording(rec, build_node_coords())
    assert len(state.sorted_party_ids) == len(rec.parties)

    mid = 3 * 3600
    walks = active_walks_at(state, mid)
    for w in walks[:5]:
        x, y = walk_position(state, w, mid)
        assert 0 <= x <= 1100
        assert 0 <= y <= 1100

    pid = state.sorted_party_ids[0]
    g = party_state_at(state, pid, mid)
    assert g is not None
    assert "status" in g


def test_visualize_window_is_scaled_down():
    from visualize import (
        CONTROL_HEIGHT,
        PARK_HEIGHT,
        PARK_LOGICAL,
        PARK_WIDTH,
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
        SIDEBAR_WIDTH,
        UI_SCALE,
        _s,
        _xy,
    )

    assert UI_SCALE == pytest.approx(0.85)
    assert PARK_WIDTH == int(PARK_LOGICAL * UI_SCALE)
    assert PARK_HEIGHT == int(PARK_LOGICAL * UI_SCALE)
    assert SIDEBAR_WIDTH == int(320 * UI_SCALE)
    assert CONTROL_HEIGHT == int(70 * UI_SCALE)
    assert SCREEN_WIDTH == PARK_WIDTH + SIDEBAR_WIDTH
    assert SCREEN_HEIGHT == PARK_HEIGHT + CONTROL_HEIGHT
    assert SCREEN_WIDTH < 1320
    assert SCREEN_HEIGHT < 1070
    assert _xy(1000, 1000) == (PARK_WIDTH, PARK_HEIGHT)
    assert _s(100) == 85
