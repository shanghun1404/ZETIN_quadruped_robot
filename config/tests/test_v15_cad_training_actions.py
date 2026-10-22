"""CPU contract checks against the ROS2 walking controller, without launching Kit."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "config", ROOT / "ros2/v14_ros2_control", ROOT / "ros2/v151_ros2_control"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from v14_ros2_control import contract
from v14_ros2_control.joint_command import JointCommandFilter
from v151_ros2_control.motion import CANONICAL, hybrid_motor_command


# Independent, measured CAD mapping in the actor's articulation order.
PHYSICAL = (
    "FL_hip_roll", "FR_hip_roll", "FL_hip_pitch", "FR_hip_pitch",
    "BR_hip_roll", "BL_hip_roll", "BR_hip_pitch", "BL_hip_pitch",
    "FL_knee", "FR_knee", "BR_knee", "BL_knee",
)
SIGN = torch.tensor([-1., -1., -1., 1., 1., 1., 1., -1., 1., -1., -1., 1.])
DEFAULT = torch.tensor([0., 0., -.55, -.55, 0., 0., -.55, -.55, 1.1, 1.1, 1.1, 1.1])
CAPS = torch.tensor([24.8, 24.8, 48., 48., 24.8, 24.8, 48., 48., 48., 48., 48., 48.])


def actions_module(monkeypatch):
    """Replace only Kit's lifetime shell; production tensor math and ROS2 stay real."""
    path = ROOT / "config/v15_cad_training_actions.py"
    assert path.is_file(), "The CAD training action implementation is missing"

    class ActionTerm:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self._asset = env.scene[cfg.asset_name]

        @property
        def device(self):
            return self._env.device

        @property
        def num_envs(self):
            return self._env.num_envs

        def reset(self, env_ids=None):
            pass

    @dataclass
    class ActionTermCfg:
        asset_name: str = "robot"
        debug_vis: bool = False
        clip: dict | None = None

    modules = {
        "isaaclab.managers.action_manager": {"ActionTerm": ActionTerm},
        "isaaclab.managers.manager_term_cfg": {"ActionTermCfg": ActionTermCfg},
        "isaaclab.utils": {"configclass": dataclass},
    }
    for name, fields in modules.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_cad_training_actions_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class PhysicalAsset:
    """An articulation boundary with a deliberately different physical joint order."""

    def __init__(self, count):
        permutation = [11, 3, 7, 0, 9, 1, 6, 4, 10, 2, 8, 5]
        self.joint_names = [PHYSICAL[index] for index in permutation]
        self.num_joints = len(self.joint_names)
        self.ids = [self.joint_names.index(name) for name in PHYSICAL]
        self.data = SimpleNamespace(
            joint_pos=torch.zeros(count, 12), joint_vel=torch.zeros(count, 12),
            # CAD default angles must never replace the actor's policy offsets.
            default_joint_pos=torch.full((count, 12), .37),
            joint_stiffness=torch.full((count, 12), 300.),
            joint_damping=torch.full((count, 12), 8.),
            joint_effort_limits=torch.zeros(count, 12),
        )
        self.data.joint_effort_limits[:, self.ids] = CAPS
        self.command = None

    def find_joints(self, names, preserve_order=False):
        assert preserve_order
        return [self.joint_names.index(name) for name in names], list(names)

    def set_policy_state(self, position, velocity):
        self.data.joint_pos[:, self.ids] = position * SIGN
        self.data.joint_vel[:, self.ids] = velocity * SIGN

    def set_joint_position_target(self, position, joint_ids=None):
        assert list(joint_ids) == self.ids
        self.command = position.clone()


def make_action(module, count=3, **cfg_fields):
    asset = PhysicalAsset(count)
    generator = torch.Generator().manual_seed(1404)
    q = DEFAULT + torch.randn(count, 12, generator=generator) * .12
    dq = torch.randn(count, 12, generator=generator) * 9
    asset.set_policy_state(q, dq)
    env = SimpleNamespace(
        scene={"robot": asset}, device="cpu", num_envs=count,
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=.005)),
    )
    cfg = module.V15CADJointPositionActionCfg(**cfg_fields)
    action = module.V15CADJointPositionAction(cfg, env)
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    return action, env, asset, q, dq


def ros2_walk_target(command_filter, q, dq):
    filtered = command_filter.step(q, dq).numpy()
    to_canonical = [contract.JOINT_NAMES.index(name) for name in CANONICAL]
    to_policy = [CANONICAL.index(name) for name in contract.JOINT_NAMES]
    result = hybrid_motor_command(1., np.zeros(12), filtered[to_canonical],
                                  q.numpy()[to_canonical], dq.numpy()[to_canonical])
    return torch.tensor(result["position"][to_policy], dtype=q.dtype)


def test_action_and_both_guards_match_real_ros2_walk_with_signed_physical_joints(monkeypatch):
    # Catches physical-order/sign leaks, wrong offsets, policy-rate LPF, and dropping either guard.
    module = actions_module(monkeypatch)
    action, _, asset, q, dq = make_action(module, max_delay_steps=0)
    filters = [JointCommandFilter() for _ in range(len(q))]
    for index, command_filter in enumerate(filters):
        command_filter.reset(q[index])
    generator = torch.Generator().manual_seed(99)
    for step in range(44):
        q = DEFAULT + .35 * torch.randn(q.shape, generator=generator)
        dq = 70 * torch.randn(dq.shape, generator=generator)
        asset.set_policy_state(q, dq)
        if step % 4 == 0:
            raw = 4 * torch.randn(q.shape, generator=generator)
            action.process_actions(raw)
            target = DEFAULT + .2 * raw
            torch.testing.assert_close(action.raw_actions, raw)
            torch.testing.assert_close(action.raw_targets, target)
            for index, command_filter in enumerate(filters):
                command_filter.set_target(target[index])
        action.apply_actions()
        expected = torch.stack([ros2_walk_target(f, q[i], dq[i]) for i, f in enumerate(filters)])
        torch.testing.assert_close(action.processed_actions, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(asset.command, expected * SIGN, rtol=2e-5, atol=2e-6)


def test_canonical_observations_preserve_actor_defaults_and_policy_order(monkeypatch):
    # Catches using CAD asset defaults or unsigned/native articulation order in the actor's 235 inputs.
    module = actions_module(monkeypatch)
    _, env, _, q, dq = make_action(module)
    torch.testing.assert_close(module.canonical_joint_pos_rel(env), q - DEFAULT)
    torch.testing.assert_close(module.canonical_joint_vel(env), dq)


def test_startup_filter_and_first_target_use_measured_position(monkeypatch):
    # Catches the default-pose startup transient before the first policy action.
    module = actions_module(monkeypatch)
    action, _, asset, q, _ = make_action(module, max_delay_steps=0)
    asset.set_policy_state(q, torch.zeros_like(q))
    torch.testing.assert_close(action.raw_targets, q)
    action.apply_actions()
    torch.testing.assert_close(asset.command, q * SIGN)


def test_episode_delay_runs_at_physics_rate_and_subset_reset_has_no_old_history(monkeypatch):
    # Catches advancing delay only on policy ticks or resetting unaffected robots.
    module = actions_module(monkeypatch)
    action, _, asset, q, _ = make_action(module, count=4, effort_guard_fraction=None, walk_guard_fraction=None)
    asset.set_policy_state(q, torch.zeros_like(q))
    action._delay_line.delay[:] = torch.tensor([0, 1, 2, 3])
    filters = [JointCommandFilter(guard_fraction=None) for _ in range(4)]
    for i, command_filter in enumerate(filters):
        command_filter.reset(q[i])
    raw = torch.full_like(q, 1.5)
    action.process_actions(raw)
    target = DEFAULT + .2 * raw
    for step in range(5):
        for i, command_filter in enumerate(filters):
            command_filter.set_target(target[i] if step >= i else q[i])
        action.apply_actions()
        expected = torch.stack([f.step(q[i], torch.zeros(12)) for i, f in enumerate(filters)])
        torch.testing.assert_close(action.processed_actions, expected)
    previous = action.processed_actions.clone()
    old_history = action._delay_line.history.clone()
    new_q = q + .29
    asset.set_policy_state(new_q, torch.zeros_like(q))
    action.reset(torch.tensor([1, 3]))
    torch.testing.assert_close(action.raw_targets[[1, 3]], new_q[[1, 3]])
    torch.testing.assert_close(action.processed_actions[[1, 3]], new_q[[1, 3]])
    torch.testing.assert_close(action.raw_actions[[1, 3]], torch.zeros(2, 12))
    torch.testing.assert_close(action.processed_actions[[0, 2]], previous[[0, 2]])
    torch.testing.assert_close(action._delay_line.history[:, [0, 2]], old_history[:, [0, 2]])
    torch.testing.assert_close(action._delay_line.history[:, [1, 3]], new_q[[1, 3]].expand(4, -1, -1))
    assert bool(((action._delay_line.delay[[1, 3]] >= 0) & (action._delay_line.delay[[1, 3]] <= 3)).all())


def test_full_and_empty_resets_resample_bounded_episode_delays(monkeypatch):
    # Catches None/slice/list reset indexing errors and missing per-episode delay sampling.
    module = actions_module(monkeypatch)
    action, _, asset, q, _ = make_action(module, count=256)
    old = action._delay_line.delay.clone()
    action.reset([])
    torch.testing.assert_close(action._delay_line.delay, old)
    asset.set_policy_state(q + .5, torch.zeros_like(q))
    action.reset()
    assert set(action._delay_line.delay.tolist()) == {0, 1, 2, 3}
    torch.testing.assert_close(action.raw_targets, q + .5)
    action.reset(slice(0, 10, 2))


def test_torque_estimate_uses_actual_drive_gains_caps_and_last_pre_step_state(monkeypatch):
    # Catches nominal-gain estimates, wrong torque signs, no saturation, and post-step recomputation.
    module = actions_module(monkeypatch)
    action, _, asset, q, dq = make_action(module, max_delay_steps=0)
    generator = torch.Generator().manual_seed(101)
    gain = 240 + 120 * torch.rand(q.shape, generator=generator)
    damping = 6.4 + 3.2 * torch.rand(q.shape, generator=generator)
    limits = CAPS * .8
    asset.data.joint_stiffness[:, asset.ids] = gain
    asset.data.joint_damping[:, asset.ids] = damping
    asset.data.joint_effort_limits[:, asset.ids] = limits
    action.process_actions(torch.full_like(q, 25.))
    action.apply_actions()
    expected = (gain * (asset.command * SIGN - q) - damping * dq).clamp(-limits, limits)
    torch.testing.assert_close(action.estimated_torque_policy, expected)
    assert bool((expected.abs() == limits).any())
    snapshot = action.estimated_torque_policy.clone()
    asset.set_policy_state(q + 1., dq - 3.)
    torch.testing.assert_close(action.estimated_torque_policy, snapshot)
    action.reset([0])
    torch.testing.assert_close(action.estimated_torque_policy[0], torch.zeros(12))
    torch.testing.assert_close(action.estimated_torque_policy[1:], snapshot[1:])


def set_constant_policy_torque(asset, torque):
    """Produce real pre-step PD torque through damping, independent of the LPF."""
    asset.data.joint_stiffness.zero_()
    asset.set_policy_state(DEFAULT.expand_as(torque), -torque / 8.)


def test_rms_updates_on_each_200hz_apply_and_uses_cached_pre_step_torque(monkeypatch):
    # Catches policy-rate accumulation, post-step recomputation, and using the old torque sample.
    module = actions_module(monkeypatch)
    action, _, asset, q, _ = make_action(module, max_delay_steps=0)
    torch.testing.assert_close(action.estimated_torque_rms_policy, torch.zeros_like(q))
    set_constant_policy_torque(asset, torch.full_like(q, 10.))
    action.apply_actions()
    # At dt=0.005 s and tau=0.5 s, a first 10 Nm sample yields 0.9975052 Nm RMS.
    torch.testing.assert_close(action.estimated_torque_rms_policy, torch.full_like(q, .9975052))
    snapshot = action.estimated_torque_rms_policy.clone()
    set_constant_policy_torque(asset, torch.full_like(q, -20.))
    action.process_actions(torch.ones_like(q))
    torch.testing.assert_close(action.estimated_torque_rms_policy, snapshot)
    action.apply_actions()
    torch.testing.assert_close(action.estimated_torque_rms_policy, torch.full_like(q, 2.2282690))


def test_rms_constant_torque_converges_at_configured_time_constant(monkeypatch):
    # Catches a wrong dt/time constant, averaging joints together, and missing physics-tick updates.
    module = actions_module(monkeypatch)
    action, _, asset, _, _ = make_action(module, count=2, max_delay_steps=0, rms_time_constant_s=.25)
    torque = torch.tensor([[2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.],
                          [13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2.]])
    set_constant_policy_torque(asset, torque)
    for _ in range(50):
        action.apply_actions()
    # After one time constant, RMS is sqrt(1-exp(-1)) = 0.7950601 times the load.
    torch.testing.assert_close(action.estimated_torque_rms_policy, torque * .7950601)
    for _ in range(450):
        action.apply_actions()
    # After ten time constants it has converged to within 0.01% of constant torque.
    torch.testing.assert_close(action.estimated_torque_rms_policy, torque, rtol=1e-4, atol=1e-5)


def test_rms_short_pulse_does_not_equal_sustained_rated_torque_excess(monkeypatch):
    # Catches replacing RMS with instantaneous torque or preserving a pulse as a lasting maximum.
    module = actions_module(monkeypatch)
    action, _, asset, q, _ = make_action(module, count=2, max_delay_steps=0)
    torque = torch.full_like(q, 20.)
    set_constant_policy_torque(asset, torque)
    action.apply_actions()
    assert bool((action.estimated_torque_policy.abs() > 18.).all())
    assert bool((action.estimated_torque_rms_policy < 18.).all())
    torque[0] = 0.
    set_constant_policy_torque(asset, torque)
    for _ in range(199):
        action.apply_actions()
    # One 5 ms pulse has decayed after 1 s, while 1 s of 20 Nm exceeds the 18 Nm rating.
    torch.testing.assert_close(action.estimated_torque_rms_policy[0], torch.full((12,), .7376021))
    torch.testing.assert_close(action.estimated_torque_rms_policy[1], torch.full((12,), 18.5974699))
    assert bool((action.estimated_torque_rms_policy[0] < 18.).all())
    assert bool((action.estimated_torque_rms_policy[1] > 18.).all())


def test_rms_squares_positive_and_negative_policy_torques_per_joint(monkeypatch):
    # Catches signed averaging, signed RMS, or mixing articulation indices into the policy order.
    module = actions_module(monkeypatch)
    action, _, asset, _, _ = make_action(module, count=2, max_delay_steps=0)
    magnitude = torch.arange(1., 13.)
    torque = torch.stack((magnitude, -magnitude))
    set_constant_policy_torque(asset, torque)
    action.apply_actions()
    torch.testing.assert_close(action.estimated_torque_policy, torque)
    torch.testing.assert_close(action.estimated_torque_rms_policy,
                               magnitude.expand_as(torque) * .09975052)


@pytest.mark.parametrize("env_ids", [[1, 3], torch.tensor([1, 3]), slice(1, 4, 2)])
def test_rms_subset_reset_discards_only_selected_episode_history(monkeypatch, env_ids):
    # Catches resetting all robots, keeping stale episode load, or broken list/tensor/slice indexing.
    module = actions_module(monkeypatch)
    action, _, asset, q, _ = make_action(module, count=4, max_delay_steps=0)
    set_constant_policy_torque(asset, torch.full_like(q, 10.))
    for _ in range(50):
        action.apply_actions()
    snapshot = action.estimated_torque_rms_policy.clone()
    action.reset(env_ids)
    torch.testing.assert_close(action.estimated_torque_rms_policy[[1, 3]], torch.zeros(2, 12))
    torch.testing.assert_close(action.estimated_torque_rms_policy[[0, 2]], snapshot[[0, 2]])
    action.apply_actions()
    torch.testing.assert_close(action.estimated_torque_rms_policy[[1, 3]], torch.full((2, 12), .9975052))
    assert bool((action.estimated_torque_rms_policy[[0, 2]] > snapshot[[0, 2]]).all())


@pytest.mark.parametrize("env_ids", [None, slice(None)])
def test_rms_full_reset_clears_every_episode_and_empty_reset_preserves_history(monkeypatch, env_ids):
    # Catches None/slice full resets missing RMS state or an empty reset clearing accumulated load.
    module = actions_module(monkeypatch)
    action, _, asset, q, _ = make_action(module, max_delay_steps=0)
    set_constant_policy_torque(asset, torch.full_like(q, 10.))
    action.apply_actions()
    snapshot = action.estimated_torque_rms_policy.clone()
    action.reset([])
    torch.testing.assert_close(action.estimated_torque_rms_policy, snapshot)
    action.reset(env_ids)
    torch.testing.assert_close(action.estimated_torque_rms_policy, torch.zeros_like(q))


@pytest.mark.parametrize("time_constant", [0., -.5, float("nan"), float("inf"), -float("inf")])
def test_invalid_rms_time_constant_fails_before_simulation(monkeypatch, time_constant):
    # Catches zero/negative smoothing times and nonfinite state updates entering training.
    module = actions_module(monkeypatch)
    with pytest.raises(ValueError, match="rms_time_constant_s"):
        make_action(module, rms_time_constant_s=time_constant)


@pytest.mark.parametrize("fields", [dict(max_delay_steps=-1), dict(max_delay_steps=11),
                                    dict(max_delay_steps=1.5), dict(effort_guard_fraction=0.),
                                    dict(effort_guard_fraction=1.1), dict(scale=float("nan"))])
def test_invalid_control_parameters_fail_before_simulation(monkeypatch, fields):
    module = actions_module(monkeypatch)
    with pytest.raises(ValueError):
        make_action(module, **fields)


def test_nonfinite_or_wrong_shape_policy_action_does_not_modify_targets(monkeypatch):
    module = actions_module(monkeypatch)
    action, _, _, _, _ = make_action(module)
    initial = action.raw_targets.clone()
    with pytest.raises(ValueError):
        action.process_actions(torch.zeros(3, 11))
    with pytest.raises(ValueError):
        action.process_actions(torch.full((3, 12), float("nan")))
    torch.testing.assert_close(action.raw_targets, initial)
