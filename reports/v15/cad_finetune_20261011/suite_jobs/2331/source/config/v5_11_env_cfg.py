"""Stage 1 V5.11 reference-reward-reset environment.

V5.6-V5.10 tuned leg-usage *within* the v5 "phase-free + grounded" paradigm:
heavy ``support_deficit``/``flight`` penalties forced >=2 feet planted at all
times, the stock smoothness penalties were disabled, forward payoff was gated,
and training ran at a near-zero curriculum speed (semantic 0.03-0.06 m/s). That
paradigm produces the stiff, synchronized "tuk-tuk" shuffle the operator wants
gone.

The natural walk in the reference video (``output_video_hpc``) was produced by
the July-6 ``guide_rough`` config, which is essentially the stock IsaacLab
velocity reward: high velocity tracking, the standard smoothness penalties kept,
NO grounding constraints, and a real walking command speed (semantic 0.5-1.0
m/s). A phased trot emerges naturally there.

V5.11 reproduces that reference recipe on the v5 flat / proprioception / smooth
infrastructure. The single conceptual change from v5 is the reward + command
paradigm: strip the grounding stack, restore the stock penalties and high
tracking, add the standard ``feet_air_time`` phasing incentive (weight 0.25),
command a real walking speed, and target the reference 0.45 m stance. The robot
asset, GRU policy, proprioception observations, and smooth action filter are
unchanged.
"""

from __future__ import annotations

import isaaclab.managers as mg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from guide_frame import guide_forward_speed_range_to_base
from smooth_env_cfg import GroundedGuideEnvCfg


# Reference walking command (guide_rough): semantic forward 0.5-1.0 m/s.
V5_11_SPEED_RANGE = (0.5, 1.0)
V5_11_YAW_RATE_RANGE = (-0.5, 0.5)
# Reference stance height (guide_rough targeted 0.45 m, not the v5 low 0.35 m).
V5_11_TARGET_HEIGHT_M = 0.45

# The full reference reward table. This is the guide_rough recipe with the one
# operator-endorsed change: feet_air_time enabled at the standard IsaacLab
# weight (0.25) to give an explicit phasing incentive.
V5_11_REWARD_WEIGHTS = {
    "track_lin_vel_xy_exp": 5.00,
    "track_ang_vel_z_exp": 2.50,
    "lin_vel_z_l2": -1.00,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -1.0e-5,
    "dof_acc_l2": -1.0e-8,
    "action_rate_l2": -1.0e-3,
    "feet_air_time": 0.25,
    "undesired_contacts": -1.00,
    "flat_orientation_l2": -0.50,
    "dof_pos_limits": -0.10,
    "base_height_l2": -2.00,
}


@configclass
class EmptyCurriculumCfg:
    """No curriculum: fixed reward weights and a fixed walking command speed.

    Disabling the curriculum removes both the v5 reward-weight ramps and the
    ``velocity_command_curriculum_v5`` term that otherwise overwrites the
    command speed range every step with the near-zero semantic schedule.
    """


@configclass
class ReferenceResetGuideEnvCfg(GroundedGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Replace the entire v5 grounded reward stack with the reference
        # recipe. Reassigning to a fresh stock RewardsCfg drops every grounded
        # term (signed_forward_progress, support_deficit, flight, foot_impact,
        # foot_slip, swing_clearance, filtered_target_rate/jerk, joint_power,
        # usage gates) in one move, leaving only the stock task + penalty terms.
        self.rewards = RewardsCfg()

        self.rewards.track_lin_vel_xy_exp.weight = 5.00
        self.rewards.track_lin_vel_xy_exp.params["command_name"] = "base_velocity"
        self.rewards.track_ang_vel_z_exp.weight = 2.50
        self.rewards.track_ang_vel_z_exp.params["command_name"] = "base_velocity"

        self.rewards.lin_vel_z_l2.weight = -1.00
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -1.0e-8
        self.rewards.action_rate_l2.weight = -1.0e-3

        # Standard phasing incentive: reward each foot for air time between
        # contacts. This is the IsaacLab-native lever that produces a phased
        # stepping gait; the v5 line disabled it.
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_foot"
        )
        self.rewards.feet_air_time.params["command_name"] = "base_velocity"

        self.rewards.undesired_contacts.weight = -1.00
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_thigh|.*_calf"
        )
        self.rewards.flat_orientation_l2.weight = -0.50
        self.rewards.dof_pos_limits.weight = -0.10

        # Base-height feedback toward the reference 0.45 m stance.
        self.rewards.base_height_l2 = mg.RewardTermCfg(
            func=mdp.base_height_l2,
            weight=-2.00,
            params={
                "target_height": V5_11_TARGET_HEIGHT_M,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        # --- Disable every curriculum term (fixed weights + fixed speed).
        self.curriculum = EmptyCurriculumCfg()

        # --- Command a real walking speed like the reference, so a dynamic
        # phased gait is actually required (v5 trained near standstill).
        self.commands.base_velocity.ranges.lin_vel_x = guide_forward_speed_range_to_base(
            V5_11_SPEED_RANGE
        )
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = V5_11_YAW_RATE_RANGE
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
