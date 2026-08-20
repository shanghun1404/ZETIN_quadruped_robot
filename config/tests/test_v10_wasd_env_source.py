"""Runtime-boundary tests for the V10 flat-WASD Isaac configuration."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_flat_wasd_config_preserves_blind_48_to_12_parent_and_wires_commands(monkeypatch):
    pytest.importorskip("torch")

    def module(name: str) -> ModuleType:
        value = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("isaaclab")
    module("isaaclab.envs")
    module("isaaclab.envs.mdp")
    commands_boundary = module("isaaclab.envs.mdp.commands")
    utils_boundary = module("isaaclab.utils")

    class UniformVelocityCommand:
        pass

    class UniformVelocityCommandCfg:
        class Ranges:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    commands_boundary.UniformVelocityCommand = UniformVelocityCommand
    commands_boundary.UniformVelocityCommandCfg = UniformVelocityCommandCfg
    utils_boundary.configclass = lambda cls: cls

    gait_reward = type("CommandTrackingGaitReward", (), {})

    def linear_reward(*args, **kwargs):
        raise AssertionError("reward execution is outside config construction")

    reward_module = module("v13_mdp")
    reward_module.CommandTrackingGaitReward = gait_reward
    reward_module.command_aware_base_linear_velocity_reward = linear_reward

    parent_module = module("v10_3_flat_env_cfg")

    class GuideV2FlatSpawnFixedEnvCfg:
        def __init__(self):
            self.policy_observation_width = 48
            self.action_width = 12
            self.curriculum = None
            self.scene = SimpleNamespace(
                height_scanner=None,
                terrain=SimpleNamespace(
                    terrain_type="plane",
                    terrain_generator=None,
                ),
            )
            self.commands = SimpleNamespace(base_velocity=None)
            self.rewards = SimpleNamespace(
                gait=SimpleNamespace(func=object(), params={}),
                base_linear_velocity=SimpleNamespace(
                    func=object(),
                    params={
                        "std": 1.0,
                        "ramp_at_vel": 1.0,
                        "ramp_rate": 0.5,
                        "asset_cfg": SimpleNamespace(name="robot"),
                    },
                ),
            )

        def __post_init__(self):
            return None

    parent_module.GuideV2FlatSpawnFixedEnvCfg = GuideV2FlatSpawnFixedEnvCfg

    ppo_parent = module("v10_3_flat_ppo_cfg")
    ppo_parent.V10_3_FLAT_PPO_CFG = SimpleNamespace(
        experiment_name="v10_3",
        run_name="flat",
        actor_obs_width=48,
        action_width=12,
    )

    for name in ("v10_wasd_commands", "v10_wasd_env_cfg", "v10_wasd_ppo_cfg"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    command_module = importlib.import_module("v10_wasd_commands")
    env_module = importlib.import_module("v10_wasd_env_cfg")
    ppo_module = importlib.import_module("v10_wasd_ppo_cfg")

    cfg = env_module.GuideV2V10WASDEnvCfg()
    cfg.__post_init__()

    assert issubclass(
        command_module.WASDVelocityCommand,
        UniformVelocityCommand,
    )
    assert command_module.WASDVelocityCommandCfg.class_type is command_module.WASDVelocityCommand
    assert cfg.policy_observation_width == 48
    assert cfg.action_width == 12
    assert cfg.curriculum is None
    assert cfg.scene.height_scanner is None
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert cfg.commands.base_velocity.resampling_time_range == (4.0, 4.0)
    assert cfg.commands.base_velocity.heading_command is False
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-0.3, 0.3)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (-0.5, 0.5)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (-0.4, 0.4)
    assert cfg.rewards.gait.func is gait_reward
    assert cfg.rewards.gait.params["standing_std"] == 0.05
    assert cfg.rewards.base_linear_velocity.func is linear_reward
    assert cfg.rewards.base_linear_velocity.params["std"] == 1.0
    assert cfg.rewards.base_linear_velocity.params["standing_std"] == 0.08
    assert ppo_module.V10_WASD_PPO_CFG.experiment_name == "guide_v10_wasd_flat_robot_v2"
    assert ppo_module.V10_WASD_PPO_CFG.run_name == "robot_v2_flat_wasd_demo"
    assert ppo_module.V10_WASD_PPO_CFG.actor_obs_width == 48
    assert ppo_module.V10_WASD_PPO_CFG.action_width == 12
