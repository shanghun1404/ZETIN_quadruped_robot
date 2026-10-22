from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.managers as mg
from isaaclab.utils import configclass

from guide_frame import guide_forward_speed_range_to_base
from smooth_env_cfg import GroundedGuideEnvCfg, RAMPED_TERMS
from stance_calibration import TARGET_HEIGHT_M
from v5_3_schedule import semantic_speed_range_v5_3
import v5_3_mdp


V5_3_REWARD_WEIGHTS = {
    "track_lin_vel_xy_exp": 3.00,
    "track_ang_vel_z_exp": 0.75,
    "base_height_centered": -0.50,
    "flat_orientation_l2": -2.50,
    "lateral_velocity": -2.00,
    "action_rate_l2": -0.01,
    "dof_pos_limits": -2.00,
    "undesired_contacts": -0.01,
}


@configclass
class ReferenceTrainingGuideEnvCfg(GroundedGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Preserve the V5.2 action term and observations while bounding motion
        # to the small residual range used by the stable Figure8 reference.
        self.actions.joint_pos.scale = 0.20
        self.commands.base_velocity.ranges.lin_vel_x = guide_forward_speed_range_to_base(
            semantic_speed_range_v5_3(0)
        )

        self.rewards.track_lin_vel_xy_exp.weight = V5_3_REWARD_WEIGHTS[
            "track_lin_vel_xy_exp"
        ]
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.12
        self.rewards.track_ang_vel_z_exp.weight = V5_3_REWARD_WEIGHTS[
            "track_ang_vel_z_exp"
        ]
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.50
        self.rewards.base_height_centered.weight = V5_3_REWARD_WEIGHTS[
            "base_height_centered"
        ]
        self.rewards.base_height_centered.params["target"] = TARGET_HEIGHT_M
        self.rewards.flat_orientation_l2.weight = V5_3_REWARD_WEIGHTS[
            "flat_orientation_l2"
        ]
        self.rewards.lateral_velocity.weight = V5_3_REWARD_WEIGHTS[
            "lateral_velocity"
        ]
        self.rewards.action_rate_l2 = mg.RewardTermCfg(
            func=base_mdp.action_rate_l2,
            weight=V5_3_REWARD_WEIGHTS["action_rate_l2"],
        )
        self.rewards.dof_pos_limits.weight = V5_3_REWARD_WEIGHTS[
            "dof_pos_limits"
        ]
        self.rewards.undesired_contacts.weight = V5_3_REWARD_WEIGHTS[
            "undesired_contacts"
        ]

        self.rewards.signed_forward_progress = None
        self.rewards.wrong_direction = None
        self.rewards.support_deficit = None
        self.rewards.flight = None
        self.rewards.yaw_rate = None
        self.rewards.heading_hold = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.vertical_acceleration = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.foot_impact = None
        self.rewards.foot_slip = None
        self.rewards.swing_clearance = None
        self.rewards.filtered_target_rate = None
        self.rewards.filtered_target_jerk = None
        self.rewards.joint_power = None
        self.rewards.termination = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.feet_air_time = None

        for term_name in RAMPED_TERMS:
            setattr(self.curriculum, f"ramp_{term_name}", None)
        self.curriculum.velocity_commands = mg.CurriculumTermCfg(
            func=v5_3_mdp.velocity_command_curriculum_v5_3
        )
