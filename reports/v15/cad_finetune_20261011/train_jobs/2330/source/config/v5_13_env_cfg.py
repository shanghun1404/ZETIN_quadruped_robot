"""Stage 1 V5.13: V5.11 reference reward + measured-cheat penalties.

V5.13 keeps the one V5.12 penalty that demonstrably worked (``abandoned_foot``
lifted the rear-left contact fraction 0.361 -> 0.410) and replaces the one that
did not (``pair_synchrony``, whose event-based form left the front-pair HFE
correlation at +0.68) with the dense, loophole-free ``pair_in_phase`` term.

Penalty budget: both terms are bounded to [0, 1] and masked while standing, so
the worst case is -2.0 + -1.5 = -3.5 per step against a forward-tracking reward
of up to +5.0. Walking imperfectly always beats standing still.
"""

from __future__ import annotations

import isaaclab.managers as mg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import v5_12_mdp
import v5_13_mdp
from v5_11_env_cfg import ReferenceResetGuideEnvCfg, V5_11_REWARD_WEIGHTS


PAIR_IN_PHASE_WEIGHT = -2.00
ABANDONED_FOOT_WEIGHT = -1.50
VEL_SCALE = 4.0

V5_13_REWARD_WEIGHTS = {
    **V5_11_REWARD_WEIGHTS,
    "pair_in_phase": PAIR_IN_PHASE_WEIGHT,
    "abandoned_foot": ABANDONED_FOOT_WEIGHT,
}


@configclass
class MeasuredAntiCheatGuideEnvCfg(ReferenceResetGuideEnvCfg):
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

        # Cheat 1 (measured): the two front legs swinging as one unit.
        self.rewards.pair_in_phase = mg.RewardTermCfg(
            func=v5_13_mdp.pair_in_phase,
            weight=-2.00,
            params={
                "asset_cfg": ordered_hfe,
                "command_name": "base_velocity",
                "moving_threshold": 0.05,
                "vel_scale": 4.0,
            },
        )

        # Cheat 2: one leg abandoned. Kept unchanged from V5.12, where it worked.
        self.rewards.abandoned_foot = mg.RewardTermCfg(
            func=v5_12_mdp.abandoned_foot_penalty,
            weight=-1.50,
            params={
                "threshold": 5.0,
                "sensor_cfg": ordered_sensor,
                "command_name": "base_velocity",
                "moving_threshold": 0.05,
                "window_s": 1.0,
                "contact_floor": 0.35,
            },
        )
