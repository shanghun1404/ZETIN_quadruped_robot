"""CPU checks for ramped push, motor-delay and sensor-noise randomization."""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "config", ROOT / "ros2/v14_ros2_control", ROOT / "ros2/v151_ros2_control"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from test_v15_cad_training_actions import DEFAULT, SIGN, PhysicalAsset  # noqa: E402


def robust_module(monkeypatch):
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
    loaded = {}
    for name in ("v15_cad_training_actions", "v15_cad_robust"):
        spec = importlib.util.spec_from_file_location(name, ROOT / f"config/{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["v15_cad_robust"]


def make_robust_action(module, settings, count=64):
    asset = PhysicalAsset(count)
    q = DEFAULT + .1 * torch.randn(count, 12, generator=torch.Generator().manual_seed(7))
    asset.set_policy_state(q, torch.zeros_like(q))
    env = SimpleNamespace(scene={"robot": asset}, device="cpu", num_envs=count,
                          cfg=SimpleNamespace(sim=SimpleNamespace(dt=.005)), v15_robust_settings=settings)
    action = module.V15CADRobustJointPositionAction(module.V15CADRobustJointPositionActionCfg(), env)
    return action, env


def test_ramp_runs_from_adopted_training_values_to_end(monkeypatch):
    # Catches a ramp that starts harsher than the adopted model's 0-15 ms delay or never reaches the end.
    m = robust_module(monkeypatch)
    assert m.ramp_level(0, 100) == 0 and m.ramp_level(50, 100) == .5 and m.ramp_level(500, 100) == 1
    assert m.ramp_level(0, 0) == 1
    assert m.interpolate_settings(0.) == m.ROBUST_START
    assert m.interpolate_settings(1.) == m.ROBUST_END
    assert m.ROBUST_START.delay_steps == (0, 3)
    middle = m.interpolate_settings(.5)
    assert middle.push_speed_mps == pytest.approx((0., .65))
    assert middle.delay_steps[1] in (4, 5) and middle.hold_probability == pytest.approx(.025)


def test_settings_reject_impossible_values(monkeypatch):
    m = robust_module(monkeypatch)
    for bad in (dict(push_speed_mps=(1., .5)), dict(delay_steps=(0, 11)), dict(delay_steps=(0, 2.5)),
                dict(hold_probability=1.), dict(noise_scale=-.1)):
        with pytest.raises(ValueError):
            m.RobustSettings(**bad)


def test_noise_touches_only_proprioception_within_bounds(monkeypatch):
    # Catches noise on commands, last actions or the depth-derived scan, and unbounded noise.
    m = robust_module(monkeypatch)
    obs = torch.randn(512, 235, generator=torch.Generator().manual_seed(3))
    noisy = m.corrupt_proprioception(obs, 2., generator=torch.Generator().manual_seed(4))
    delta = noisy - obs
    width = 2. * torch.tensor(m.PROPRIOCEPTIVE_NOISE)
    assert bool((delta[:, :48].abs() <= width + 1e-6).all())
    assert bool((delta[:, 9:12] == 0).all()) and bool((delta[:, 36:] == 0).all())
    for block in (slice(0, 3), slice(3, 6), slice(6, 9), slice(12, 24), slice(24, 36)):
        assert float(delta[:, block].abs().max()) > .8 * float(width[block].max())
    assert m.corrupt_proprioception(obs, 0.) is obs
    with pytest.raises(ValueError):
        m.corrupt_proprioception(obs[:, :48], 1.)


def test_push_speed_and_heading(monkeypatch):
    m = robust_module(monkeypatch)
    fixed = m.planar_push_velocity(1000, (1., 1.), generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(torch.linalg.vector_norm(fixed, dim=-1), torch.ones(1000))
    heading = torch.atan2(fixed[:, 1], fixed[:, 0])
    assert float(heading.min()) < -.9 * math.pi and float(heading.max()) > .9 * math.pi
    ranged = torch.linalg.vector_norm(m.planar_push_velocity(1000, (0., .3)), dim=-1)
    assert float(ranged.max()) <= .3 + 1e-6


def test_reset_draws_delay_from_current_settings_only_for_reset_robots(monkeypatch):
    # Catches keeping the 0-3 step delay after the ramp widened it, or redrawing unaffected robots.
    m = robust_module(monkeypatch)
    action, env = make_robust_action(m, m.RobustSettings(delay_steps=(6, 6)))
    assert action.cfg.max_delay_steps == 6
    action.reset()
    assert bool((action._delay_line.delay == 6).all())
    env.v15_robust_settings = m.RobustSettings(delay_steps=(0, 0))
    action.reset(torch.tensor([1, 5]))
    assert action._delay_line.delay.tolist().count(0) == 2 and int(action._delay_line.delay[0]) == 6
    env.v15_robust_settings = m.RobustSettings(delay_steps=(0, 8))
    with pytest.raises(ValueError):
        action.reset()


def test_missed_update_keeps_previous_target_for_held_robots(monkeypatch):
    # Catches holding the raw action (seen by the actor) instead of the motor target, or holding everyone.
    m = robust_module(monkeypatch)
    action, env = make_robust_action(m, m.RobustSettings(hold_probability=.5))
    action.reset()
    previous = action.raw_targets.clone()
    raw = torch.full((64, 12), 1.5)
    torch.manual_seed(11)
    action.process_actions(raw)
    torch.testing.assert_close(action.raw_actions, raw)
    new = DEFAULT + .2 * raw
    held = (action.raw_targets == previous).all(-1)
    updated = (action.raw_targets == new).all(-1)
    assert bool((held | updated).all()) and 10 < int(held.sum()) < 54
    env.v15_robust_settings = m.RobustSettings()
    action.process_actions(raw)
    torch.testing.assert_close(action.raw_targets, new.expand(64, 12))


def test_missing_settings_fail_loudly(monkeypatch):
    m = robust_module(monkeypatch)
    action, env = make_robust_action(m, m.RobustSettings())
    del env.v15_robust_settings
    with pytest.raises(RuntimeError):
        action.reset()


def test_resumed_segment_continues_the_ramp(monkeypatch):
    # Catches a resumed segment restarting at the light start settings.
    m = robust_module(monkeypatch)
    env = SimpleNamespace(common_step_counter=100)
    assert m.robust_ramp(env, None, ramp_steps=1000, offset_steps=400) == pytest.approx(.5)
    assert env.v15_robust_settings == m.interpolate_settings(.5)
    env.common_step_counter = 0
    assert m.robust_ramp(env, None, ramp_steps=1000, offset_steps=2000) == 1
    assert env.v15_robust_settings == m.ROBUST_END


def test_motor_features_off_reproduce_parent_commands_exactly(monkeypatch):
    # Catches the robust WALK override drifting from the parent action when jitter, feedback delay and lag are off.
    m = robust_module(monkeypatch)
    robust, env = make_robust_action(m, m.RobustSettings(delay_steps=(0, 3)), count=4)
    asset = env.scene["robot"]
    twin = PhysicalAsset(4)
    twin.data.joint_pos.copy_(asset.data.joint_pos)
    twin.data.joint_vel.copy_(asset.data.joint_vel)
    parent = m.V15CADJointPositionAction(m.V15CADJointPositionActionCfg(),
                                         SimpleNamespace(scene={"robot": twin}, device="cpu", num_envs=4,
                                                         cfg=SimpleNamespace(sim=SimpleNamespace(dt=.005))))
    robust.reset()
    parent._delay_line.delay[:] = robust._delay_line.delay
    generator = torch.Generator().manual_seed(21)
    for step in range(48):
        q = DEFAULT + .35 * torch.randn(4, 12, generator=generator)
        dq = 60 * torch.randn(4, 12, generator=generator)
        asset.set_policy_state(q, dq)
        twin.set_policy_state(q, dq)
        if step % 4 == 0:
            raw = 4 * torch.randn(4, 12, generator=generator)
            robust.process_actions(raw)
            parent.process_actions(raw)
        robust.apply_actions()
        parent.apply_actions()
        assert torch.equal(asset.command, twin.command)
        assert torch.equal(robust.processed_actions, parent.processed_actions)
        assert torch.equal(robust.estimated_torque_policy, parent.estimated_torque_policy)
    assert torch.equal(robust.estimated_torque_rms_policy, parent.estimated_torque_rms_policy)
    assert torch.equal(robust.observed_joint_state()[0], robust.joint_pos_policy)


def test_motor_lag_is_first_order_and_drive_target_reproduces_lagged_torque(monkeypatch):
    # Catches applying the PD torque instantly, lagging the position target instead of torque, or nominal-gain conversion.
    m = robust_module(monkeypatch)
    action, env = make_robust_action(m, m.RobustSettings(delay_steps=(0, 0), motor_lag_s=(.01, .01)), count=3)
    asset = env.scene["robot"]
    asset.data.joint_stiffness[:] = 250.
    asset.data.joint_damping[:] = 7.
    asset.set_policy_state(DEFAULT.expand(3, 12), torch.full((3, 12), .4))
    action.reset()
    action.process_actions(torch.full((3, 12), .5))
    alpha = 1 - math.exp(-.005 / .01)
    q = asset.data.joint_pos[:, asset.ids]
    dq = asset.data.joint_vel[:, asset.ids]
    caps = asset.data.joint_effort_limits[:, asset.ids]
    lagged = torch.zeros(3, 12)
    for _ in range(6):
        action.apply_actions()
        requested = 250. * (action.processed_actions * SIGN - q) - 7. * dq
        lagged = lagged + alpha * (torch.minimum(torch.maximum(requested, -caps), caps) - lagged)
        torch.testing.assert_close(250. * (asset.command - q) - 7. * dq, lagged, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(action.estimated_torque_policy, lagged * SIGN, rtol=1e-5, atol=1e-4)
    assert float(lagged.abs().max()) > 0


def test_command_jitter_stays_in_range_and_never_reorders_targets(monkeypatch):
    # Catches jitter that lets an older target replace a newer one or leaves the episode delay range.
    m = robust_module(monkeypatch)
    torch.manual_seed(5)
    episode = torch.tensor([0, 2, 4] * 100)
    delay = episode.clone()
    seen = set()
    for _ in range(60):
        nxt = m.next_command_delay(delay, episode, 2, 6)
        assert bool((nxt >= episode).all() and (nxt <= episode + 2).all() and (nxt <= delay + 1).all())
        seen.update((nxt - episode).tolist())
        delay = nxt
    assert seen == {0, 1, 2}
    assert int(m.next_command_delay(torch.tensor([6]), torch.tensor([6]), 2, 6)) == 6


def test_feedback_delay_feeds_guards_and_actor_with_old_state(monkeypatch):
    # Catches guards or actor observations using fresh joint state when feedback is late.
    m = robust_module(monkeypatch)
    action, env = make_robust_action(m, m.RobustSettings(delay_steps=(0, 0), feedback_delay_steps=(2, 2)), count=2)
    asset = env.scene["robot"]
    states = [DEFAULT + .05 * i for i in range(5)]
    asset.set_policy_state(states[0].expand(2, 12), torch.zeros(2, 12))
    action.reset()
    action.process_actions(torch.full((2, 12), 50.))
    for i in range(4):
        asset.set_policy_state(states[i].expand(2, 12), torch.zeros(2, 12))
        action.apply_actions()
    # At the last physics step the guards could only see states[1]; the far target saturates both guards.
    limits = torch.tensor([24.8, 24.8, 48., 48., 24.8, 24.8, 48., 48., 48., 48., 48., 48.])
    torch.testing.assert_close(action.processed_actions, (states[1] + .99 * limits / 300.).expand(2, 12))
    asset.set_policy_state(states[4].expand(2, 12), torch.zeros(2, 12))
    position, _ = action.observed_joint_state()
    torch.testing.assert_close(position, states[2].expand(2, 12))
    env.v15_robust_settings = m.RobustSettings(feedback_delay_steps=(0, 3))
    with pytest.raises(ValueError):
        action.process_actions(torch.zeros(2, 12))


def test_motor_profile_ramps_from_the_same_start(monkeypatch):
    m = robust_module(monkeypatch)
    assert m.interpolate_settings(0., m.ROBUST_START, m.MOTOR_END) == m.ROBUST_START
    assert m.interpolate_settings(1., m.ROBUST_START, m.MOTOR_END) == m.MOTOR_END
    assert m.MOTOR_END.delay_steps[1] + m.MOTOR_END.delay_jitter_steps == m.ROBUST_END.delay_steps[1]
    env = SimpleNamespace(common_step_counter=10_000)
    assert m.robust_ramp(env, None, ramp_steps=100, profile="motor") == 1
    assert env.v15_robust_settings == m.MOTOR_END
    with pytest.raises(ValueError):
        m.RobustSettings(motor_lag_s=(.02, .01))
