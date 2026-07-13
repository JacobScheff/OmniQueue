"""Training and model smoke tests."""

import torch
import torch.optim as optim

from model import ParkRouterModel, obs_flat_to_tensors
from training.checkpoint import default_model, load_checkpoint, save_checkpoint
from training.features import ENV_DYNAMIC_FEAT_DIM, FLAT_OBS_DIM, GUEST_FEAT_DIM, NUM_ACTIONS, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM
from training.ppo_train import Agent, _compute_gae


def test_model_forward_shape():
    model = ParkRouterModel(
        guest_feat_dim=GUEST_FEAT_DIM,
        num_rides=NUM_RIDES,
        ride_dynamic_feat_dim=RIDE_DYNAMIC_FEAT_DIM,
        environment_dynamic_feat_dim=ENV_DYNAMIC_FEAT_DIM,
    )
    guest = torch.randn(2, 1, GUEST_FEAT_DIM)
    ride = torch.randn(2, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    env = torch.randn(2, ENV_DYNAMIC_FEAT_DIM)
    logits, value = model(guest, ride, env)
    assert logits.shape == (2, 1, NUM_ACTIONS)
    assert value.shape == (2, 1)


def test_obs_flat_to_tensors():
    flat = torch.randn(4, FLAT_OBS_DIM)
    guest, ride, env = obs_flat_to_tensors(flat)
    assert guest.shape == (4, 1, GUEST_FEAT_DIM)
    assert ride.shape == (4, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
    assert env.shape == (4, ENV_DYNAMIC_FEAT_DIM)


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

    obs = torch.randn(8, FLAT_OBS_DIM)
    _, logprob, _, value = agent.get_action_and_value(obs)
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
    # With λ=1, γ=0.5, V=0: last δ = 1 + 0.5*10 - 0 = 6
    assert abs(float(adv[-1]) - 6.0) < 1e-5
    assert abs(float(ret[-1]) - 6.0) < 1e-5

    dones_term = torch.tensor([0.0, 0.0, 1.0])
    adv_t, _ = _compute_gae(
        rewards, values, dones_term, gamma=0.5, gae_lambda=1.0, bootstrap_value=10.0
    )
    # Terminal last step ignores bootstrap: δ = 1 + 0 - 0 = 1
    assert abs(float(adv_t[-1]) - 1.0) < 1e-5
