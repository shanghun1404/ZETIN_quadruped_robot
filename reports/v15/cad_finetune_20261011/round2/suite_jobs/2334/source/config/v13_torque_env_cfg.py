"""V12 depth policy with only action smoothing and actuator-cost changes."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

import v13_torque_mdp
from v11_terrain_env_cfg import GuideV2ScannerTerrainEnvCfg, V11_REWARD_WEIGHTS
from v12_camera import make_v12_depth_camera_cfg
from v13_torque_actions import V13LowPassJointPositionActionCfg


V13_REWARD_WEIGHTS = {
    **V11_REWARD_WEIGHTS,
    "action_smoothness": -2.5,
    "torque_pressure": -0.2,
    "joint_power": -0.2,
}


@configclass
class GuideV13TorqueEnvCfg(GuideV2ScannerTerrainEnvCfg):
    """Keep V12 robot, terrain, commands and observations; regularize actuation."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = V13LowPassJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.2,
            use_default_offset=True,
            cutoff_hz=25.0,
        )
        self.scene.depth_camera = make_v12_depth_camera_cfg(self.decimation * self.sim.dt)

        self.rewards.action_smoothness.func = v13_torque_mdp.policy_action_rate_squared
        self.rewards.action_smoothness.weight = V13_REWARD_WEIGHTS["action_smoothness"]
        joints = SceneEntityCfg("robot", joint_names=".*")
        self.rewards.torque_pressure = RewardTermCfg(
            func=v13_torque_mdp.requested_torque_pressure,
            weight=V13_REWARD_WEIGHTS["torque_pressure"],
            params={"asset_cfg": joints},
        )
        self.rewards.joint_power = RewardTermCfg(
            func=v13_torque_mdp.applied_joint_power,
            weight=V13_REWARD_WEIGHTS["joint_power"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
