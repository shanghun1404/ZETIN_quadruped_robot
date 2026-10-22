"""V14 self-collision curriculum with V13 torque guard and V11 gait rewards."""

from __future__ import annotations

import copy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from v11_terrain_env_cfg import V11_REWARD_WEIGHTS
from v13_torque_env_cfg import GuideV13TorqueEnvCfg
from v14_terrain_filter import apply_v14_ground_filter


V14_REWARD_WEIGHTS = {
    **V11_REWARD_WEIGHTS,
    "action_smoothness": V11_REWARD_WEIGHTS["action_smoothness"],
    "torque_pressure": 0.0,
    "joint_power": 0.0,
}


@configclass
class GuideV14LegEnvCfg(GuideV13TorqueEnvCfg):
    """Train around inter-leg contacts while preserving the torque request guard."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.effort_guard_fraction = 0.99
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.replicate_physics = False
        self.events.v14_ground_filter = EventTerm(
            func=apply_v14_ground_filter, mode="prestartup"
        )
        self.rewards.action_smoothness.weight = V11_REWARD_WEIGHTS["action_smoothness"]
        self.rewards.torque_pressure = None
        self.rewards.joint_power = None


@configclass
class GuideV14StairEnvCfg(GuideV14LegEnvCfg):
    """Concentrate later PPO updates on stair ascent and descent."""

    def __post_init__(self):
        super().__post_init__()
        generator = copy.deepcopy(self.scene.terrain.terrain_generator)
        if generator is None:
            raise ValueError("V14 stair adaptation requires generated terrain")
        proportions = {
            "pyramid_stairs_inv": 0.45,
            "pyramid_stairs": 0.25,
            "boxes": 0.10,
            "random_rough": 0.10,
            "hf_pyramid_slope": 0.05,
            "hf_pyramid_slope_inv": 0.05,
        }
        if set(generator.sub_terrains) != set(proportions):
            raise ValueError("unexpected V14 terrain generator mix")
        for name, proportion in proportions.items():
            generator.sub_terrains[name].proportion = proportion
        self.scene.terrain.terrain_generator = generator


@configclass
class GuideV14DirectionEnvCfg(GuideV14StairEnvCfg):
    """Train the existing observation interface to follow corrective yaw commands."""

    def __post_init__(self):
        super().__post_init__()
        from v14_heading_command import make_lane_command_cfg

        self.commands.base_velocity = make_lane_command_cfg(self.commands.base_velocity)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.rewards.base_angular_velocity.params["std"] = 0.25
        self.rewards.base_angular_velocity.weight = 7.5
