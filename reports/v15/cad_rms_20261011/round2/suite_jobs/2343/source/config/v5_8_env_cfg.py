from __future__ import annotations

import isaaclab.managers as mg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import v5_8_mdp
from stance_calibration import TARGET_HEIGHT_M
from v5_6_env_cfg import PhaseDiverseGuideEnvCfg


# V5.8 reward table. Identical to V5.6 except the variance-based
# ``foot_usage_imbalance`` term is removed and the stock ``track_lin_vel_xy_exp``
# forward reward is replaced by the usage-gated ``usage_gated_forward_velocity``
# term carrying the same 3.00 weight.
V5_8_REWARD_WEIGHTS = {
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
    "usage_gated_forward_velocity": 3.00,
    "foot_slip": -0.10,
    "swing_clearance": -0.05,
}

# Gate tuning (inlined as literals in the reward params below so the training
# contract can verify them): a worst-foot contact duty below the 0.35 contact
# floor shrinks the forward reward linearly with gain 1.5. The 0.0 ``min_gate``
# is the residual gate for a fully abandoned leg and removes the forward payoff
# entirely; raise it to 0.05 if a fresh policy collapses to a stationary stand.


@configclass
class GatedVelocityGuideEnvCfg(PhaseDiverseGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_centered.params["target"] = TARGET_HEIGHT_M

        # V5.8 isolated change: drop the V5.6 variance imbalance term and replace
        # the stock forward-velocity reward with a usage-gated version. The gate
        # closes when any foot's contact duty drops below GATE_CONTACT_FLOOR, so
        # the three-leg cheat can no longer collect forward-tracking reward. Every
        # other V5.6 variable (GRU + CNN policy, action scale 0.15, actuator,
        # URDF, command curriculum, 0.35 m target height, and all remaining reward
        # terms and weights) is preserved.
        self.rewards.foot_usage_imbalance = None

        std = self.rewards.track_lin_vel_xy_exp.params["std"]
        command_name = self.rewards.track_lin_vel_xy_exp.params.get(
            "command_name", "base_velocity"
        )
        self.rewards.track_lin_vel_xy_exp = None

        ordered_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_sensor = SceneEntityCfg(
            "contact_forces",
            body_names=ordered_feet,
            preserve_order=True,
        )
        self.rewards.usage_gated_forward_velocity = mg.RewardTermCfg(
            func=v5_8_mdp.usage_gated_forward_velocity,
            weight=V5_8_REWARD_WEIGHTS["usage_gated_forward_velocity"],
            params={
                "std": std,
                "threshold": 5.0,
                "sensor_cfg": ordered_sensor,
                "command_name": command_name,
                "moving_threshold": 0.05,
                "window_s": 1.0,
                "contact_floor": 0.35,
                "gain": 1.5,
                "min_gate": 0.0,
            },
        )
