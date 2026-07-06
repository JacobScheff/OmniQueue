"""Training and model smoke tests."""

import torch

from model import ParkRouterModel, obs_flat_to_tensors
from training.features import ENV_DYNAMIC_FEAT_DIM, FLAT_OBS_DIM, GUEST_FEAT_DIM, NUM_ACTIONS, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM


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
