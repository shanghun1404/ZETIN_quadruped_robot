from __future__ import annotations

import isaaclab.managers as mg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import smooth_mdp
import v5_4_mdp
from stance_calibration import TARGET_HEIGHT_M
from v5_3_env_cfg import ReferenceTrainingGuideEnvCfg


V5_4_REWARD_WEIGHTS = {
    "track_lin_vel_xy_exp": 3.00,
    "track_ang_vel_z_exp": 0.75,
    "base_height_centered": -0.50,
    "flat_orientation_l2": -2.50,
    "lateral_velocity_raw_l2": -2.00,
    "action_rate_l2": -0.01,
    "dof_pos_limits": -2.00,
    "undesired_contacts": -0.01,
    "yaw_rate": -0.25,
    "heading_hold": -0.25,
    "vertical_velocity": -0.20,
    "roll_pitch_rate": -0.35,
    "flight": -2.00,
    "foot_impact": -0.10,
}


@configclass
class PatternNeutralGuideEnvCfg(ReferenceTrainingGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.scale = 0.15
        self.rewards.track_lin_vel_xy_exp.weight = V5_4_REWARD_WEIGHTS[
            "track_lin_vel_xy_exp"
        ]
        self.rewards.track_ang_vel_z_exp.weight = V5_4_REWARD_WEIGHTS[
            "track_ang_vel_z_exp"
        ]
        self.rewards.base_height_centered.weight = V5_4_REWARD_WEIGHTS[
            "base_height_centered"
        ]
        self.rewards.base_height_centered.params["target"] = TARGET_HEIGHT_M
        self.rewards.flat_orientation_l2.weight = V5_4_REWARD_WEIGHTS[
            "flat_orientation_l2"
        ]
        self.rewards.action_rate_l2.weight = V5_4_REWARD_WEIGHTS["action_rate_l2"]
        self.rewards.dof_pos_limits.weight = V5_4_REWARD_WEIGHTS["dof_pos_limits"]
        self.rewards.undesired_contacts.weight = V5_4_REWARD_WEIGHTS[
            "undesired_contacts"
        ]

        self.rewards.lateral_velocity = None
        self.rewards.lateral_velocity_raw_l2 = mg.RewardTermCfg(
            func=v5_4_mdp.guide_lateral_velocity_raw_l2,
            weight=V5_4_REWARD_WEIGHTS["lateral_velocity_raw_l2"],
        )
        self.rewards.yaw_rate = mg.RewardTermCfg(
            func=smooth_mdp.yaw_rate_normalized,
            weight=V5_4_REWARD_WEIGHTS["yaw_rate"],
            params={"scale": 0.15},
        )
        self.rewards.heading_hold = mg.RewardTermCfg(
            func=smooth_mdp.heading_hold_penalty,
            weight=V5_4_REWARD_WEIGHTS["heading_hold"],
        )
        self.rewards.vertical_velocity = mg.RewardTermCfg(
            func=smooth_mdp.vertical_velocity_normalized,
            weight=V5_4_REWARD_WEIGHTS["vertical_velocity"],
            params={"scale": 0.20},
        )
        self.rewards.roll_pitch_rate = mg.RewardTermCfg(
            func=smooth_mdp.roll_pitch_rate_normalized,
            weight=V5_4_REWARD_WEIGHTS["roll_pitch_rate"],
            params={"scale": 0.60},
        )
        ordered_feet = SceneEntityCfg(
            "contact_forces",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            preserve_order=True,
        )
        self.rewards.flight = mg.RewardTermCfg(
            func=smooth_mdp.flight,
            weight=V5_4_REWARD_WEIGHTS["flight"],
            params={"threshold": 5.0, "sensor_cfg": ordered_feet},
        )
        self.rewards.foot_impact = mg.RewardTermCfg(
            func=smooth_mdp.foot_impact_normalized,
            weight=V5_4_REWARD_WEIGHTS["foot_impact"],
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=".*_foot",
                )
            },
        )
