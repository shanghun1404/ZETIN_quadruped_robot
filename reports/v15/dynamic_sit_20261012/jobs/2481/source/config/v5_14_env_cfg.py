"""Stage 1 V5.14: fix the posture that forces the front-pair synchrony.

Two changes on top of V5.13, both aimed at the *posture* root cause rather than
at the phase symptom:

1. **A working posture penalty.** V5.11-V5.13 carried ``base_height_l2`` at
   weight -2.0, but that term is a raw squared error: at the measured 10.4 cm
   height deficit it produced ``2.0 * 0.104^2 = 0.022`` per step against a
   forward reward of up to 5.0 -- about 0.4%. Raising the weight does not help
   (``-6.0`` still yields only 0.065). The posture objective was effectively
   absent, which is why the robot has walked crouched since V5.11. V5.14
   replaces it with the bounded ``base_height_centered`` used by V5.6-V5.10,
   whose output saturates at 1.0 so the configured weight is the real per-step
   cost.

2. **A swing floor.** ``overplanted_foot`` presses the forefeet (planted ~84% of
   the time) to leave the ground. This is the mirror of ``abandoned_foot``, the
   one V5.12 penalty that moved its metric.

``pair_in_phase`` and ``abandoned_foot`` are retained at reduced weight: the
rear pair is already out of phase and the minimum contact fraction already
clears its floor, so they now only need to hold those gains.

Penalty budget: the three movement-masked penalties total 4.0 and the always-on
posture penalty is 2.0, against a forward reward of up to 5.0. Standing still is
not a refuge because the posture penalty applies while stationary too.
"""

from __future__ import annotations

import isaaclab.managers as mg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import smooth_mdp
import v5_12_mdp
import v5_13_mdp
import v5_14_mdp
from v5_11_env_cfg import ReferenceResetGuideEnvCfg, V5_11_REWARD_WEIGHTS, V5_11_TARGET_HEIGHT_M


PAIR_IN_PHASE_WEIGHT = -1.00
ABANDONED_FOOT_WEIGHT = -1.00
OVERPLANTED_FOOT_WEIGHT = -2.00
BASE_HEIGHT_CENTERED_WEIGHT = -2.00
SWING_FLOOR = 0.25
HEIGHT_SCALE = 0.10

V5_14_REWARD_WEIGHTS = {
    key: value for key, value in V5_11_REWARD_WEIGHTS.items() if key != "base_height_l2"
}
V5_14_REWARD_WEIGHTS.update(
    {
        "pair_in_phase": PAIR_IN_PHASE_WEIGHT,
        "abandoned_foot": ABANDONED_FOOT_WEIGHT,
        "overplanted_foot": OVERPLANTED_FOOT_WEIGHT,
        "base_height_centered": BASE_HEIGHT_CENTERED_WEIGHT,
    }
)


@configclass
class PostureFixGuideEnvCfg(ReferenceResetGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        ordered_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_sensor = SceneEntityCfg(
            "contact_forces", body_names=ordered_feet, preserve_order=True
        )
        ordered_hfe = SceneEntityCfg(
            "robot",
            joint_names=["FL_HFE", "FR_HFE", "RL_HFE", "RR_HFE"],
            preserve_order=True,
        )

        # The raw squared-error height term never applied real pressure.
        self.rewards.base_height_l2 = None
        self.rewards.base_height_centered = mg.RewardTermCfg(
            func=smooth_mdp.base_height_centered,
            weight=-2.00,
            params={"target": 0.45, "scale": 0.10},
        )

        # New main lever: force the over-planted forefeet to lift.
        self.rewards.overplanted_foot = mg.RewardTermCfg(
            func=v5_14_mdp.overplanted_foot_penalty,
            weight=-2.00,
            params={
                "threshold": 5.0,
                "sensor_cfg": ordered_sensor,
                "command_name": "base_velocity",
                "moving_threshold": 0.05,
                "window_s": 1.0,
                "swing_floor": 0.25,
            },
        )

        # Hold the V5.13 gains.
        self.rewards.pair_in_phase = mg.RewardTermCfg(
            func=v5_13_mdp.pair_in_phase,
            weight=-1.00,
            params={
                "asset_cfg": ordered_hfe,
                "command_name": "base_velocity",
                "moving_threshold": 0.05,
                "vel_scale": 4.0,
            },
        )
        self.rewards.abandoned_foot = mg.RewardTermCfg(
            func=v5_12_mdp.abandoned_foot_penalty,
            weight=-1.00,
            params={
                "threshold": 5.0,
                "sensor_cfg": ordered_sensor,
                "command_name": "base_velocity",
                "moving_threshold": 0.05,
                "window_s": 1.0,
                "contact_floor": 0.35,
            },
        )
