from __future__ import annotations

import isaaclab.managers as mg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import v5_7_mdp
from stance_calibration import TARGET_HEIGHT_M
from v5_6_env_cfg import PhaseDiverseGuideEnvCfg


V5_7_REWARD_WEIGHTS = {
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
    "foot_impact": -0.05,
    "support_deficit": -2.00,
    "all_feet_stance": -0.35,
    "contact_transition_burst": -0.75,
    "per_foot_usage_floor": -0.75,
    "foot_slip": -0.10,
    "swing_clearance": -0.05,
}


@configclass
class UsageFloorGuideEnvCfg(PhaseDiverseGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_centered.params["target"] = TARGET_HEIGHT_M
        self.rewards.foot_usage_imbalance = None
        ordered_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_sensor = SceneEntityCfg(
            "contact_forces",
            body_names=ordered_feet,
            preserve_order=True,
        )
        self.rewards.per_foot_usage_floor = mg.RewardTermCfg(
            func=v5_7_mdp.per_foot_usage_floor,
            weight=V5_7_REWARD_WEIGHTS["per_foot_usage_floor"],
            params={
                "threshold": 5.0,
                "sensor_cfg": ordered_sensor,
                "window_s": 1.0,
                "contact_floor": 0.35,
                "swing_floor": 0.10,
            },
        )
