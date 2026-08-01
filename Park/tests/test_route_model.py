"""Tests for multi-ride route decoder, shaping, and CF helpers."""

import numpy as np
import torch

from Park.model import ParkRouterModel, forward_route_with_mask
from Park.training.checkpoint import default_model, load_checkpoint, save_checkpoint
from Park.training.features import (
    D_MODEL,
    ENV_DYNAMIC_FEAT_DIM,
    GUEST_FEAT_DIM,
    NUM_ACTIONS,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    RIDE_FEAT_HISTORY,
    RIDE_FEAT_MUST_DO,
    RIDE_FEAT_OPEN,
    RIDE_FEAT_WALK,
    ROUTE_PAD,
    js_divergence,
    rewrite_prefs_must_dos,
    top_must_do_or_pref,
)
from Park.training.route_reward import (
    consistency_bonus,
    pad_route,
    planned_walk_penalty,
    realized_walk_penalty,
)


def _open_obs(batch: int = 2):
    guest = torch.zeros(batch, GUEST_FEAT_DIM)
    guest[..., 37] = 0.5
    guest[..., :NUM_RIDES] = 1.0 / NUM_RIDES
    ride = torch.zeros(batch, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    ride[..., RIDE_FEAT_OPEN] = 1.0
    ride[..., RIDE_FEAT_WALK] = 0.1
    env = torch.zeros(batch, ENV_DYNAMIC_FEAT_DIM)
    return guest, ride, env


def test_route_decode_shape_and_no_repeat():
    model = default_model("cpu")
    guest, ride, env = _open_obs(4)
    out = forward_route_with_mask(model, guest, ride, env, deterministic=True)
    assert out.routes.shape == (4, model.route_k)
    assert out.log_prob.shape == (4,)
    assert out.entropy.shape == (4,)
    assert out.slot0_logits.shape == (4, NUM_ACTIONS)
    for b in range(4):
        rides = [int(a) for a in out.routes[b].tolist() if 0 <= int(a) < NUM_RIDES]
        assert len(rides) == len(set(rides))


def test_exit_pads_route():
    model = default_model("cpu")
    guest, ride, env = _open_obs(1)
    # Soft-close → exit only
    env[..., 0] = 1.0
    guest[..., 37] = 0.0
    out = forward_route_with_mask(model, guest, ride, env, deterministic=True)
    assert int(out.routes[0, 0].item()) == NUM_RIDES  # exit
    assert all(int(x.item()) == ROUTE_PAD for x in out.routes[0, 1:])


def test_teacher_force_logprob_finite():
    model = default_model("cpu")
    guest, ride, env = _open_obs(3)
    with torch.no_grad():
        sampled = forward_route_with_mask(model, guest, ride, env, deterministic=False)
    out = forward_route_with_mask(
        model, guest, ride, env, routes=sampled.routes, deterministic=False
    )
    assert torch.isfinite(out.log_prob).all()
    assert torch.isfinite(out.entropy).all()


def test_slot0_forward_still_works():
    model = ParkRouterModel(
        guest_feat_dim=GUEST_FEAT_DIM,
        num_rides=NUM_RIDES,
        ride_dynamic_feat_dim=RIDE_DYNAMIC_FEAT_DIM,
        environment_dynamic_feat_dim=ENV_DYNAMIC_FEAT_DIM,
        d_model=D_MODEL,
        route_k=6,
    )
    guest, ride, env = _open_obs(2)
    logits, value = model(guest, ride, env)
    assert logits.shape == (2, NUM_ACTIONS)
    assert value.shape == (2, 1)


def test_consistency_and_walk_shaping():
    prev = pad_route([1, 2, 3, 4, 5, 6])
    new = pad_route([2, 3, 9, 10, 11, 12])  # matches shift on first two
    bonus = consistency_bonus(new, prev)
    assert bonus > 0.0
    pen = planned_walk_penalty(new)
    assert pen >= 0.0
    assert realized_walk_penalty(600.0) > 0.0


def test_rewrite_prefs_changes_must_dos():
    guest, ride, env = _open_obs(1)
    ride[..., 0, RIDE_FEAT_HISTORY] = 1.0
    g2, r2 = rewrite_prefs_must_dos(guest, ride)
    assert not torch.allclose(guest[:, :NUM_RIDES], g2[:, :NUM_RIDES])
    # Completed ride cannot be a must-do
    assert float(r2[0, 0, RIDE_FEAT_MUST_DO].item()) == 0.0
    top = top_must_do_or_pref(g2, r2)
    assert 0 <= int(top[0].item()) < NUM_RIDES


def test_js_divergence_identical_zero():
    p = torch.softmax(torch.randn(2, 5), dim=-1)
    assert float(js_divergence(p, p).max().item()) < 1e-5


def test_checkpoint_partial_load(tmp_path):
    donor = default_model("cpu")
    path = tmp_path / "route.pt"
    save_checkpoint(path, donor, None, step=3, extra={"phase": "ppo"})
    loaded, step, extra = load_checkpoint(path, "cpu")
    assert step == 3
    assert extra.get("arch_version") == "route_v1"
    guest, ride, env = _open_obs(1)
    out = forward_route_with_mask(loaded, guest, ride, env, deterministic=True)
    assert out.routes.shape[1] == loaded.route_k
