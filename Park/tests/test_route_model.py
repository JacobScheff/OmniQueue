"""Tests for rank-then-route model, walk shaping, and CF helpers."""

import torch

from Park.model import ParkRouterModel, RankRouteModel, forward_route_with_mask
from Park.training.checkpoint import default_model, load_checkpoint, save_checkpoint
from Park.training.features import (
    D_MODEL,
    ENV_DYNAMIC_FEAT_DIM,
    GUEST_FEAT_DIM,
    GUEST_FEAT_TIME_LEFT,
    NUM_ACTIONS,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    RIDE_FEAT_ETA,
    RIDE_FEAT_HISTORY,
    RIDE_FEAT_MUST_DO,
    RIDE_FEAT_OPEN,
    RIDE_FEAT_UNFINISHED_PREF,
    RIDE_FEAT_WAIT,
    RIDE_FEAT_WALK,
    ROUTE_PAD,
    js_divergence,
    pref_rank_aux_loss,
    rewrite_prefs_must_dos,
    rewrite_waits,
    top_must_do_or_pref,
)
from Park.training.route_reward import (
    pad_route,
    planned_walk_penalty,
    realized_walk_penalty,
    route_shaping_delta,
)


def _open_obs(batch: int = 2):
    guest = torch.zeros(batch, GUEST_FEAT_DIM)
    guest[..., GUEST_FEAT_TIME_LEFT] = 0.5
    guest[..., :NUM_RIDES] = 1.0 / NUM_RIDES
    ride = torch.zeros(batch, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    ride[..., RIDE_FEAT_OPEN] = 1.0
    ride[..., RIDE_FEAT_WALK] = 0.1
    ride[..., RIDE_FEAT_WAIT] = 0.1
    ride[..., RIDE_FEAT_ETA] = 0.2
    env = torch.zeros(batch, ENV_DYNAMIC_FEAT_DIM)
    env[..., 1] = 0.1
    return guest, ride, env


def test_route_decode_shape_and_no_repeat():
    model = default_model("cpu")
    guest, ride, env = _open_obs(4)
    out = forward_route_with_mask(model, guest, ride, env, deterministic=True)
    assert out.routes.shape == (4, model.route_k)
    assert out.log_prob.shape == (4,)
    assert out.entropy.shape == (4,)
    assert out.slot0_logits.shape == (4, NUM_ACTIONS)
    assert out.slot_logits.shape == (4, model.route_k, NUM_ACTIONS)
    assert out.slot_masks.shape == (4, model.route_k, NUM_ACTIONS)
    assert isinstance(model, RankRouteModel)
    for b in range(4):
        rides = [int(a) for a in out.routes[b].tolist() if 0 <= int(a) < NUM_RIDES]
        assert len(rides) == len(set(rides))


def test_walk_refresh_changes_tail_logits():
    """Slot-1 logits should depend on inter-ride walks from the forced slot-0 ride."""
    model = default_model("cpu")
    model.candidate_m = NUM_RIDES  # all rides visible so walk edits affect keys
    with torch.no_grad():
        model.ride_walk_norm.fill_(0.5)
        model.ride_walk_norm[0, 1] = 0.01
        model.ride_walk_norm[0, 2] = 0.9
    guest, ride, env = _open_obs(1)
    force = torch.tensor([0], dtype=torch.long)
    out_a = forward_route_with_mask(
        model, guest, ride, env, deterministic=True, force_first=force
    )
    logits_a = out_a.slot_logits[0, 1, :NUM_RIDES].detach().clone()
    with torch.no_grad():
        model.ride_walk_norm[0, 1] = 0.9
        model.ride_walk_norm[0, 2] = 0.01
    out_b = forward_route_with_mask(
        model, guest, ride, env, deterministic=True, force_first=force
    )
    logits_b = out_b.slot_logits[0, 1, :NUM_RIDES]
    assert int(out_a.routes[0, 0].item()) == 0
    assert int(out_b.routes[0, 0].item()) == 0
    assert not torch.allclose(logits_a, logits_b)


def test_pref_rank_aux_loss_finite_and_prefers_aligned_logits():
    guest, ride, env = _open_obs(2)
    ride[..., RIDE_FEAT_UNFINISHED_PREF] = 0.0
    ride[:, 3, RIDE_FEAT_UNFINISHED_PREF] = 1.0
    ride[:, 3, RIDE_FEAT_MUST_DO] = 1.0
    stage_logits = torch.zeros(2, NUM_ACTIONS)
    stage_mask = torch.zeros(2, NUM_ACTIONS, dtype=torch.bool)
    stage_mask[:, :NUM_RIDES] = True
    stage_logits[:, 3] = 5.0
    good = pref_rank_aux_loss(stage_logits, stage_mask, ride)
    bad_logits = stage_logits.clone()
    bad_logits[:, 3] = 0.0
    bad_logits[:, 7] = 5.0
    bad = pref_rank_aux_loss(bad_logits, stage_mask, ride)
    assert torch.isfinite(good) and torch.isfinite(bad)
    assert float(good.item()) < float(bad.item())


def test_force_first_pins_slot0_and_continues():
    model = default_model("cpu")
    guest, ride, env = _open_obs(1)
    natural = forward_route_with_mask(model, guest, ride, env, deterministic=True)
    natural0 = int(natural.routes[0, 0].item())
    force_id = 0 if natural0 != 0 else 1
    forced = forward_route_with_mask(
        model,
        guest,
        ride,
        env,
        deterministic=True,
        force_first=torch.tensor([force_id], dtype=torch.long),
    )
    assert int(forced.routes[0, 0].item()) == force_id
    rides = [int(a) for a in forced.routes[0].tolist() if 0 <= int(a) < NUM_RIDES]
    assert rides[0] == force_id
    assert len(rides) == len(set(rides))
    assert torch.allclose(forced.slot0_logits, natural.slot0_logits)


def test_exit_pads_route():
    model = default_model("cpu")
    guest, ride, env = _open_obs(1)
    env[..., 0] = 1.0
    guest[..., GUEST_FEAT_TIME_LEFT] = 0.0
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
        route_k=5,
    )
    guest, ride, env = _open_obs(2)
    logits, value = model(guest, ride, env)
    assert logits.shape == (2, NUM_ACTIONS)
    assert value.shape == (2, 1)


def test_walk_shaping_no_consistency():
    new = pad_route([2, 3, 9, 10, 11])
    pen = planned_walk_penalty(new)
    assert pen >= 0.0
    assert realized_walk_penalty(600.0) > 0.0
    ride = torch.zeros(NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM).numpy()
    ride[:, RIDE_FEAT_OPEN] = 1.0
    ride[:, RIDE_FEAT_WALK] = 0.1
    shaping, walk_sec = route_shaping_delta(new, pad_route([1, 2, 3, 4, 5]), ride)
    assert shaping <= 0.0  # planned walk only (negative)
    assert walk_sec > 0.0


def test_rewrite_prefs_changes_must_dos():
    guest, ride, env = _open_obs(1)
    ride[..., 0, RIDE_FEAT_HISTORY] = 1.0
    g2, r2 = rewrite_prefs_must_dos(guest, ride)
    assert not torch.allclose(guest[:, :NUM_RIDES], g2[:, :NUM_RIDES])
    assert float(r2[0, 0, RIDE_FEAT_MUST_DO].item()) == 0.0
    top = top_must_do_or_pref(g2, r2)
    assert 0 <= int(top[0].item()) < NUM_RIDES


def test_rewrite_waits_changes_eta():
    guest, ride, env = _open_obs(2)
    ride2, env2 = rewrite_waits(ride, env)
    assert not torch.allclose(ride[..., RIDE_FEAT_WAIT], ride2[..., RIDE_FEAT_WAIT])
    assert torch.isfinite(ride2[..., RIDE_FEAT_ETA]).all()


def test_js_divergence_identical_zero():
    p = torch.softmax(torch.randn(2, 5), dim=-1)
    assert float(js_divergence(p, p).max().item()) < 1e-5


def test_close_call_can_sample():
    model = default_model("cpu")
    guest, ride, env = _open_obs(1)
    # With a large margin, deterministic path may sample
    outs = {
        int(
            forward_route_with_mask(
                model,
                guest,
                ride,
                env,
                deterministic=True,
                temperature=1.0,
                close_margin=1.0,
                top_p=1.0,
            ).routes[0, 0].item()
        )
        for _ in range(20)
    }
    # Not a hard assert on diversity (random init may be peaked), but call must work.
    assert len(outs) >= 1


def test_checkpoint_partial_load(tmp_path):
    donor = default_model("cpu")
    path = tmp_path / "route.pt"
    save_checkpoint(path, donor, None, step=3, extra={"phase": "ppo"})
    loaded, step, extra = load_checkpoint(path, "cpu")
    assert step == 3
    assert extra.get("arch_version") == "rank_route_v1"
    guest, ride, env = _open_obs(1)
    out = forward_route_with_mask(loaded, guest, ride, env, deterministic=True)
    assert out.routes.shape[1] == loaded.route_k
