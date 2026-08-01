"""Training and model smoke tests."""

import numpy as np
import torch
import torch.optim as optim

from Park.model import ParkRouterModel, forward_with_mask, obs_flat_to_tensors
from Park.training.checkpoint import TrainConfig, default_model, load_checkpoint, save_checkpoint
from Park.training.features import (
    D_MODEL,
    ENV_DYNAMIC_FEAT_DIM,
    FLAT_OBS_DIM,
    GUEST_FEAT_DIM,
    NUM_ACTIONS,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    RIDE_FEAT_OPEN,
    RIDE_FEAT_WALK,
    apply_action_mask,
    build_action_mask,
    masked_cross_entropy,
)
from Park.training.ppo_train import Agent, _compute_gae


def test_model_forward_shape():
    model = ParkRouterModel(
        guest_feat_dim=GUEST_FEAT_DIM,
        num_rides=NUM_RIDES,
        ride_dynamic_feat_dim=RIDE_DYNAMIC_FEAT_DIM,
        environment_dynamic_feat_dim=ENV_DYNAMIC_FEAT_DIM,
        d_model=D_MODEL,
    )
    guest = torch.randn(2, GUEST_FEAT_DIM)
    ride = torch.randn(2, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = torch.randn(2, ENV_DYNAMIC_FEAT_DIM)
    logits, value = model(guest, ride, env)
    assert logits.shape == (2, NUM_ACTIONS)
    assert value.shape == (2, 1)


def test_obs_flat_to_tensors():
    flat = torch.randn(4, FLAT_OBS_DIM)
    guest, ride, env = obs_flat_to_tensors(flat)
    assert guest.shape == (4, GUEST_FEAT_DIM)
    assert ride.shape == (4, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    assert env.shape == (4, ENV_DYNAMIC_FEAT_DIM)


def test_action_mask_closes_broken_and_soft_close():
    guest = torch.zeros(1, GUEST_FEAT_DIM)
    guest[..., 37] = 0.5  # time left
    ride = torch.zeros(1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    ride[..., RIDE_FEAT_OPEN] = 1.0
    ride[..., 0, RIDE_FEAT_OPEN] = 0.0  # ride 0 closed
    ride[..., RIDE_FEAT_WALK] = 0.1
    env = torch.zeros(1, ENV_DYNAMIC_FEAT_DIM)

    mask = build_action_mask(guest, ride, env)
    assert mask.shape == (1, NUM_ACTIONS)
    assert not bool(mask[0, 0])
    assert bool(mask[0, 1])
    assert bool(mask[0, NUM_RIDES])  # exit
    assert bool(mask[0, NUM_RIDES + 1])  # idle

    env[..., 0] = 1.0  # soft close
    mask_closed = build_action_mask(guest, ride, env)
    assert bool(mask_closed[0, NUM_RIDES])
    assert not bool(mask_closed[0, NUM_RIDES + 1])
    assert not bool(mask_closed[0, 1])


def test_masked_cross_entropy_ignores_illegal():
    logits = torch.zeros(2, NUM_ACTIONS)
    logits[0, 3] = 5.0
    actions = torch.tensor([3, 0])
    action_mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
    action_mask[1, :] = False
    loss = masked_cross_entropy(logits, actions, action_mask)
    assert torch.isfinite(loss)
    assert float(loss) < 1.0


def test_masked_cross_entropy_keeps_label_legal():
    logits = torch.zeros(1, NUM_ACTIONS)
    logits[0, 7] = 2.0
    actions = torch.tensor([7])
    action_mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    loss = masked_cross_entropy(logits, actions, action_mask)
    assert torch.isfinite(loss)


def test_forward_with_mask_sets_illegal_logits():
    model = default_model("cpu")
    guest = torch.zeros(1, GUEST_FEAT_DIM)
    guest[..., 37] = 0.5
    ride = torch.zeros(1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    ride[..., RIDE_FEAT_OPEN] = 1.0
    ride[..., 0, RIDE_FEAT_OPEN] = 0.0
    ride[..., RIDE_FEAT_WALK] = 0.1
    env = torch.zeros(1, ENV_DYNAMIC_FEAT_DIM)
    logits, _, mask = forward_with_mask(model, guest, ride, env)
    assert not bool(mask[0, 0])
    assert logits[0, 0] <= -1.0e9


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.d_model == 256
    assert cfg.ride_dynamic_feat_dim == 8


def test_ppo_warm_start_keeps_optimizer_linked(tmp_path):
    """BC init must load into Agent.model in-place so Adam still tracks live params."""
    device = torch.device("cpu")
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3, eps=1e-5)

    donor = default_model(device)
    with torch.no_grad():
        for p in donor.parameters():
            p.fill_(0.25)
    ckpt = tmp_path / "bc_final.pt"
    save_checkpoint(ckpt, donor, None, step=7, extra={"phase": "bc"})

    loaded, step, extra = load_checkpoint(ckpt, device)
    assert step == 7
    assert extra.get("phase") == "bc"
    agent.model.load_state_dict(loaded.state_dict())

    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    agent_ids = {id(p) for p in agent.parameters()}
    assert opt_ids == agent_ids

    flat = torch.randn(8, FLAT_OBS_DIM)
    g_end = GUEST_FEAT_DIM
    r_end = g_end + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    flat[:, 37] = 0.5
    ride_view = flat[:, g_end:r_end].view(-1, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    ride_view[..., RIDE_FEAT_OPEN] = 1.0
    ride_view[..., RIDE_FEAT_WALK] = 0.1

    _, logprob, _, value = agent.get_action_and_value(flat)
    loss = -(logprob.mean() + value.mean())
    optimizer.zero_grad()
    loss.backward()
    before = {n: p.detach().clone() for n, p in agent.named_parameters()}
    optimizer.step()
    changed = sum(1 for n, p in agent.named_parameters() if not torch.equal(before[n], p))
    assert changed > 0


def test_compute_gae_bootstraps_when_truncated():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.0, 0.0, 0.0])
    dones = torch.tensor([0.0, 0.0, 0.0])  # truncated, not terminal
    adv, ret = _compute_gae(
        rewards, values, dones, gamma=0.5, gae_lambda=1.0, bootstrap_value=10.0
    )
    assert abs(float(adv[-1]) - 6.0) < 1e-5
    assert abs(float(ret[-1]) - 6.0) < 1e-5

    dones_term = torch.tensor([0.0, 0.0, 1.0])
    adv_t, _ = _compute_gae(
        rewards, values, dones_term, gamma=0.5, gae_lambda=1.0, bootstrap_value=10.0
    )
    assert abs(float(adv_t[-1]) - 1.0) < 1e-5


def test_apply_action_mask_finite():
    logits = torch.zeros(2, NUM_ACTIONS)
    mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
    mask[0, 0] = False
    out = apply_action_mask(logits, mask)
    assert out[0, 0] <= -1.0e9
    assert torch.isfinite(out).all()
