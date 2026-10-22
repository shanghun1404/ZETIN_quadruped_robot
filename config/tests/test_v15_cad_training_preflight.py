"""Reject cleaned divergence and check the installed RSL-RL observation API."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

from v15_cad_training_preflight import validate_preflight_state


def state():
    robot = SimpleNamespace(data=SimpleNamespace(
        root_state_w=torch.zeros(2, 13), joint_pos=torch.zeros(2, 12),
        joint_vel=torch.zeros(2, 12)))
    obs = TensorDict({'policy': torch.zeros(2, 235)}, batch_size=[2])
    return robot, obs


def test_native_actor_accepts_observation_groups_and_rejects_bare_policy_tensor():
    robot, obs = state()
    policy = ActorCritic(obs, {'policy': ['policy'], 'critic': ['policy']}, 12,
                        actor_hidden_dims=[16], critic_hidden_dims=[16])
    assert policy.act_inference(obs).shape == (2, 12)
    with pytest.raises((IndexError, TypeError)):
        policy.act_inference(obs['policy'])
    validate_preflight_state(0, robot.data, obs, torch.zeros(2))


def test_clean_observations_cannot_hide_a_nonfinite_wrapper_event():
    robot, obs = state()
    with pytest.raises(RuntimeError, match='sanitized'):
        validate_preflight_state(1, robot.data, obs, torch.zeros(2))


@pytest.mark.parametrize('field', ['root_state_w', 'joint_pos', 'joint_vel'])
def test_raw_state_is_checked_even_with_finite_observations(field):
    robot, obs = state()
    getattr(robot.data, field)[0, 0] = float('nan')
    with pytest.raises(RuntimeError, match=field):
        validate_preflight_state(0, robot.data, obs, torch.zeros(2))


def test_divergent_episode_reset_cannot_hide_bad_physics():
    robot, obs = state()
    with pytest.raises(RuntimeError, match='physics divergence'):
        validate_preflight_state(0, robot.data, obs, torch.zeros(2),
                                 torch.tensor([False, True]))
