from __future__ import annotations

import isaaclab.managers as mg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import smooth_mdp
import v5_6_mdp
from stance_calibration import TARGET_HEIGHT_M
from v5_4_env_cfg import PatternNeutralGuideEnvCfg


V5_6_REWARD_WEIGHTS = {
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
    "foot_usage_imbalance": -0.30,
    "foot_slip": -0.10,
    "swing_clearance": -0.05,
}


@configclass
class PhaseDiverseGuideEnvCfg(PatternNeutralGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_centered.params["target"] = TARGET_HEIGHT_M
        self.rewards.foot_impact.weight = V5_6_REWARD_WEIGHTS["foot_impact"]

        ordered_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_sensor = SceneEntityCfg(
            "contact_forces",
            body_names=ordered_feet,
            preserve_order=True,
        )
        ordered_asset = SceneEntityCfg(
            "robot",
            body_names=ordered_feet,
            preserve_order=True,
        )
        self.rewards.support_deficit = mg.RewardTermCfg(
            func=smooth_mdp.support_deficit,
            weight=V5_6_REWARD_WEIGHTS["support_deficit"],
            params={"threshold": 5.0, "sensor_cfg": ordered_sensor},
        )
        self.rewards.all_feet_stance = mg.RewardTermCfg(
            func=v5_6_mdp.all_feet_stance,
            weight=V5_6_REWARD_WEIGHTS["all_feet_stance"],
            params={"threshold": 5.0, "sensor_cfg": ordered_sensor},
        )
        self.rewards.contact_transition_burst = mg.RewardTermCfg(
            func=v5_6_mdp.contact_transition_burst,
            weight=V5_6_REWARD_WEIGHTS["contact_transition_burst"],
            params={"sensor_cfg": ordered_sensor},
        )
        self.rewards.foot_usage_imbalance = mg.RewardTermCfg(
            func=v5_6_mdp.foot_usage_imbalance,
            weight=V5_6_REWARD_WEIGHTS["foot_usage_imbalance"],
            params={
                "threshold": 5.0,
                "sensor_cfg": ordered_sensor,
                "window_s": 1.0,
            },
        )
        self.rewards.foot_slip = mg.RewardTermCfg(
            func=smooth_mdp.foot_slip_normalized,
            weight=V5_6_REWARD_WEIGHTS["foot_slip"],
            params={
                "threshold": 5.0,
                "sensor_cfg": ordered_sensor,
                "asset_cfg": ordered_asset,
            },
        )
        self.rewards.swing_clearance = mg.RewardTermCfg(
            func=smooth_mdp.swing_clearance_normalized,
            weight=V5_6_REWARD_WEIGHTS["swing_clearance"],
            params={
                "asset_cfg": ordered_asset,
                "contact_sensor_cfg": ordered_sensor,
                "height_sensor_cfg": SceneEntityCfg("height_scanner"),
                "threshold": 5.0,
                "low": 0.03,
                "high": 0.05,
                "margin": 0.03,
            },
        )
