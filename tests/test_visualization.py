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
