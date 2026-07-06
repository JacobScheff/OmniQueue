"""Tests for TorchScript export and native LibTorch rollout."""

from __future__ import annotations

import pytest
import torch

from model import ParkRouterModel
from training.features import FLAT_OBS_DIM
from training.policy_export import PolicyTorchScript, export_model_torchscript
from simulator import native_backend_name


def test_policy_torchscript_matches_eager():
    model = ParkRouterModel(
        guest_feat_dim=45,
        num_rides=35,
        ride_dynamic_feat_dim=5,
        environment_dynamic_feat_dim=4,
    )
    wrapper = PolicyTorchScript(model).eval()
    obs = torch.randn(4, FLAT_OBS_DIM)
    with torch.no_grad():
        eager_logits, eager_value = wrapper(obs)
    scripted = torch.jit.script(wrapper)
    with torch.no_grad():
        script_logits, script_value = scripted(obs)
    assert torch.allclose(eager_logits, script_logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(eager_value, script_value, atol=1e-5, rtol=1e-4)


def test_export_model_torchscript_roundtrip(tmp_path):
    model = ParkRouterModel(
        guest_feat_dim=45,
        num_rides=35,
        ride_dynamic_feat_dim=5,
        environment_dynamic_feat_dim=4,
    )
    out = tmp_path / "policy.ts.pt"
    export_model_torchscript(model, out)
    loaded = torch.jit.load(str(out))
    obs = torch.randn(2, FLAT_OBS_DIM)
    with torch.no_grad():
        logits, value = loaded(obs)
    assert logits.shape == (2, 37)
    assert value.shape == (2,)


@pytest.mark.skipif(native_backend_name() != "native", reason="C++ extension not built")
def test_native_ppo_rollout_smoke(tmp_path):
    import _park_sim

    if not getattr(_park_sim, "HAS_NATIVE_PPO_ROLLOUT", False):
        pytest.skip("LibTorch rollout not enabled in _park_sim")

    model = ParkRouterModel(
        guest_feat_dim=45,
        num_rides=35,
        ride_dynamic_feat_dim=5,
        environment_dynamic_feat_dim=4,
    )
    policy_path = tmp_path / "policy.ts.pt"
    export_model_torchscript(model, policy_path)

    result = _park_sim.collect_ppo_rollout(str(policy_path), 7, 256, False, "cpu")
    assert result.total_steps > 1000
    assert result.n == 256
    assert len(result.advantages) == 256
    assert result.metrics.rides_completed > 0
