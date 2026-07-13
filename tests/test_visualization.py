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


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_prefetch_and_arc_cache():
    from park_graph import get_park_graph, reset_park_graph
    from visualize import (
        MAX_FRAME_DT,
        MAX_WALK_DOTS,
        PREFETCH_UNTIL_SEC,
        ReplayState,
        build_node_coords,
        prefetch_walk_polylines,
        walk_position,
    )

    assert MAX_WALK_DOTS <= 1500
    assert MAX_FRAME_DT <= 1.0 / 20.0
    assert PREFETCH_UNTIL_SEC >= 10 * 60

    reset_park_graph()
    rec = record_day(seed=7, sample_interval_sec=300)
    state = ReplayState.from_recording(rec, build_node_coords())
    n = prefetch_walk_polylines(state, until_sec=5 * 60, persist=True)
    assert n > 0
    park = get_park_graph()
    assert len(park._path_polylines) >= n

    # Cached entries are (polyline, cum_lengths, total)
    sample = next(iter(park._path_polylines.values()))
    poly, cum, total = sample
    assert len(poly) >= 1
    assert len(cum) == len(poly)
    assert total >= 0.0

    early = next(w for w in state.walks if float(w.start_sec) <= 5 * 60)
    x, y = walk_position(state, early, float(early.start_sec) + 1.0)
    assert 0 <= x <= 1100
    assert 0 <= y <= 1100

    # Disk cache round-trip
    n_disk = len(park._path_polylines)
    reset_park_graph()
    park2 = get_park_graph()
    assert park2.load_polyline_cache() >= n_disk
    assert len(park2._path_polylines) >= n_disk


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
