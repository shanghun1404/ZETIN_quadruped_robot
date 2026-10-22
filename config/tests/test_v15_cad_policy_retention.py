"""Real CPU ActorCritic/PPO checks for selective V15 actor retention."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[2]


def retention_module():
    path = ROOT / "config/v15_cad_policy_retention.py"
    assert path.is_file(), "Selective native PPO retention implementation is missing"
    spec = importlib.util.spec_from_file_location("_v15_cad_policy_retention_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def observations(commands):
    policy = torch.zeros(len(commands), 235)
    policy[:, 9:12] = torch.tensor(commands)
    return TensorDict({"policy": policy}, batch_size=[len(commands)])


def actor(obs, constant=None):
    model = ActorCritic(
        obs, {"policy": ["policy"], "critic": ["policy"]}, 12,
        actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=.3,
        actor_obs_normalization=True, critic_obs_normalization=True,
    )
    if constant is not None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.std.fill_(.3)
            model.actor[-1].bias.fill_(constant)
    return model


def frozen_teacher(obs, constant=1.):
    teacher = actor(obs, constant)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def fill_rollout(algorithm, obs, steps=3, zero_rewards=False):
    algorithm.init_storage("rl", obs.batch_size[0], steps, obs, [12])
    with torch.no_grad():
        for step in range(steps):
            algorithm.act(obs)
            rewards = torch.zeros(obs.batch_size[0]) if zero_rewards else torch.arange(
                obs.batch_size[0], dtype=torch.float32
            ) * .1 + step * .05
            algorithm.process_env_step(obs, rewards, torch.zeros(obs.batch_size[0]), {})
        algorithm.compute_returns(obs)


def test_command_mask_retains_forward_zero_and_inclusive_boundaries_only():
    # Catches reversed command coordinates, a positive-only forward mask, and strict boundary errors.
    module = retention_module()
    obs = observations([
        [0., .35, 0.], [0., 0., 0.], [.02, .1, .02], [-.02, 0., -.02],
        [0., -.1, 0.], [0., .35, .03], [0., .35, -.03], [.03, .35, 0.],
        [-.03, .35, 0.], [.03, 0., 0.],
    ])
    teacher = frozen_teacher(obs)
    student_mean = torch.full((10, 12), 2., requires_grad=True)
    loss, fraction = module.selective_policy_retention_loss(student_mean, teacher, obs)
    torch.testing.assert_close(loss, torch.tensor(1.))
    torch.testing.assert_close(fraction, torch.tensor(.4))
    loss.backward()
    expected = torch.zeros_like(student_mean)
    expected[:4] = 1. / 24.
    torch.testing.assert_close(student_mean.grad, expected)


def test_retention_gradient_reaches_student_actor_only_and_teacher_state_is_immutable():
    # Catches teacher gradients/normalizer updates and detached student mean predictions.
    module = retention_module()
    obs = observations([[0., .35, 0.], [0., 0., 0.], [0., -.1, 0.]])
    student, teacher = actor(obs, 2.), frozen_teacher(obs)
    snapshot = {key: value.clone() for key, value in teacher.state_dict().items()}
    student_mean = student.act_inference(obs)
    loss, fraction = module.selective_policy_retention_loss(student_mean, teacher, obs)
    loss.backward()
    torch.testing.assert_close(student.actor[-1].bias.grad, torch.full((12,), 1. / 6.))
    assert student.std.grad is None
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
    torch.testing.assert_close(fraction, torch.tensor(2. / 3.))


def test_empty_eligible_subset_is_finite_differentiable_zero():
    # Catches NaN from mean(empty), a non-differentiable zero, and unwanted reverse/yaw gradients.
    module = retention_module()
    obs = observations([[0., -.1, 0.], [0., .35, .5], [.3, .35, 0.]])
    student_mean = torch.full((3, 12), 2., requires_grad=True)
    loss, fraction = module.selective_policy_retention_loss(student_mean, frozen_teacher(obs), obs)
    assert bool(torch.isfinite(loss))
    assert loss.item() == 0.
    assert fraction.item() == 0.
    loss.backward()
    torch.testing.assert_close(student_mean.grad, torch.zeros_like(student_mean))


@pytest.mark.parametrize("mode", ["no_teacher", "zero_weight"])
@pytest.mark.parametrize("clipped_value,normalize_advantage", [(True, False), (False, True)])
def test_disabled_retention_matches_native_ppo_parameters_losses_and_adaptive_learning_rate(
    mode, clipped_value, normalize_advantage,
):
    # Catches altering native clipping, advantage normalization, optimizer/learning-rate, or sampling order.
    module = retention_module()
    torch.manual_seed(1404)
    obs = observations([[0., .35, 0.], [0., 0., 0.], [0., -.1, 0.], [0., 0., .5]])
    kwargs = dict(
        num_learning_epochs=2, num_mini_batches=2, learning_rate=.003,
        use_clipped_value_loss=clipped_value, normalize_advantage_per_mini_batch=normalize_advantage,
        desired_kl=.001, schedule="adaptive",
    )
    native = PPO(actor(obs), **kwargs)
    fill_rollout(native, obs)
    retained = module.RetentionPPO(copy.deepcopy(native.policy), retention_weight=0. if mode == "zero_weight" else 5., **kwargs)
    retained.storage = copy.deepcopy(native.storage)
    if mode == "zero_weight":
        retained.retention_teacher = frozen_teacher(obs)
    torch.manual_seed(2338)
    native_losses = native.update()
    torch.manual_seed(2338)
    retained_losses = retained.update()
    for name, expected in native_losses.items():
        assert retained_losses[name] == expected
    assert retained_losses["policy_retention"] == 0.
    assert retained_losses["policy_retention_eligible_fraction"] == 0.
    assert retained.learning_rate == native.learning_rate
    assert retained.storage.step == native.storage.step == 0
    for key, value in native.policy.state_dict().items():
        torch.testing.assert_close(retained.policy.state_dict()[key], value, rtol=0, atol=0)


def test_native_rollout_update_adds_retention_before_gradient_clipping_and_freezes_teacher():
    # Catches an unused auxiliary loss, gradients added after clipping, and teacher optimization.
    module = retention_module()
    obs = observations([[0., .35, 0.], [0., 0., 0.], [0., -.1, 0.], [0., 0., .5]])
    student, teacher = actor(obs, 2.), frozen_teacher(obs)
    algorithm = module.RetentionPPO(
        student, num_learning_epochs=1, num_mini_batches=1, learning_rate=.01,
        entropy_coef=0., value_loss_coef=0., max_grad_norm=.25, schedule="fixed",
    )
    algorithm.retention_teacher = teacher
    fill_rollout(algorithm, obs, zero_rewards=True)
    snapshot = {key: value.clone() for key, value in teacher.state_dict().items()}
    with torch.no_grad():
        initial_distance = (student.act_inference(obs) - teacher.act_inference(obs)).square().mean().item()
    losses = algorithm.update()
    assert losses["policy_retention"] == pytest.approx(1.)
    assert losses["policy_retention_eligible_fraction"] == pytest.approx(.5)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    with torch.no_grad():
        final_distance = (student.act_inference(obs) - teacher.act_inference(obs)).square().mean().item()
    assert final_distance < initial_distance
    gradients = [parameter.grad.flatten() for parameter in student.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert torch.cat(gradients).norm().item() <= .250001
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
    assert algorithm.storage.step == 0


@pytest.mark.parametrize("unsupported", ["recurrent", "symmetry", "rnd", "same_teacher", "unfrozen_teacher"])
def test_active_retention_rejects_unsupported_or_mutable_teacher_contract(unsupported):
    # Catches silently applying the command mask to augmented/recurrent data or optimizing the teacher.
    module = retention_module()
    obs = observations([[0., .35, 0.]])
    algorithm = module.RetentionPPO(actor(obs))
    algorithm.retention_teacher = frozen_teacher(obs)
    if unsupported == "recurrent":
        algorithm.policy.is_recurrent = True
    elif unsupported == "symmetry":
        algorithm.symmetry = {"use_data_augmentation": True}
    elif unsupported == "rnd":
        algorithm.rnd = object()
    elif unsupported == "same_teacher":
        algorithm.retention_teacher = algorithm.policy
    else:
        algorithm.retention_teacher.requires_grad_(True)
    with pytest.raises(ValueError, match="retention"):
        algorithm.update()


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_retention_weight_must_be_finite_and_nonnegative(weight):
    # Catches a negative imitation objective or a nonfinite auxiliary loss entering native PPO.
    module = retention_module()
    obs = observations([[0., 0., 0.]])
    with pytest.raises(ValueError, match="retention_weight"):
        module.RetentionPPO(actor(obs), retention_weight=weight)
