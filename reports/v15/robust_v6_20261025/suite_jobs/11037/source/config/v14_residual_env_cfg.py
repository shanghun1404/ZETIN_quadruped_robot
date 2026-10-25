"""Pre-reset clearance and correction rewards with the original V14 gait objective."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from v14_heading_command import make_lane_command_cfg
from v14_leg_env_cfg import GuideV14StairEnvCfg
from v14_residual_math import clearance_cost
from v14_residual_policy import robot_capsule_gaps


def interleg_proximity(env):
    return clearance_cost(robot_capsule_gaps(env))


def correction_size(env):
    return (env._v14_residual_delta/env._v14_residual_max_rad).square().mean(-1)


def correction_change(env):
    return ((env._v14_residual_delta-env._v14_residual_previous)/env._v14_residual_max_rad).square().mean(-1)


@configclass
class GuideV14ResidualEnvCfg(GuideV14StairEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = make_lane_command_cfg(self.commands.base_velocity)
        self.commands.base_velocity.ranges.ang_vel_z = (-.4, .4)
        self.rewards.v14_clearance = RewardTermCfg(func=interleg_proximity, weight=-2.)
        self.rewards.v14_correction = RewardTermCfg(func=correction_size, weight=-.25)
        self.rewards.v14_correction_change = RewardTermCfg(func=correction_change, weight=-.05)
