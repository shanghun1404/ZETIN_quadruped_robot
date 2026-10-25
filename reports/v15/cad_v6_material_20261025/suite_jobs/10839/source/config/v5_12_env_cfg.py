"""Stage 1 V5.12 environment: V5.11 reference reward + anti-cheat penalties.

V5.11 hit the commanded 0.50 m/s but the measured gait was two cheats stacked:
the front pair bounded in phase (HFE FL-FR correlation +0.70) and the rear-left
foot was abandoned (36% contact versus 77-82% for the other three). V5.12 keeps
the V5.11 reference reward table *exactly* and adds two bounded penalties that
price those two behaviours directly.

Budget check (why this cannot cause the standing/suicide collapse seen in the
`action_rate_l2 -0.05` and V5.9 experiments): both penalties are bounded to
[0, 1] and are masked to zero while standing, so the worst case is
`-2.0 + -1.5 = -3.5` per step against a forward-tracking reward of up to `+5.0`.
Walking imperfectly therefore always beats standing still.
"""

from __future__ import annotations

import isaaclab.managers as mg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import v5_12_mdp
from v5_11_env_cfg import ReferenceResetGuideEnvCfg, V5_11_REWARD_WEIGHTS


PAIR_SYNCHRONY_WEIGHT = -2.00
ABANDONED_FOOT_WEIGHT = -1.50
CONTACT_FLOOR = 0.35

V5_12_REWARD_WEIGHTS = {
    **V5_11_REWARD_WEIGHTS,
    "pair_synchrony": PAIR_SYNCHRONY_WEIGHT,
    "abandoned_foot": ABANDONED_FOOT_WEIGHT,
}


@configclass
class AntiCheatGuideEnvCfg(ReferenceResetGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        ordered_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_sensor = SceneEntityCfg(
            "contact_forces", body_names=ordered_feet, preserve_order=True
        )

        # Cheat 1: the two front (or two rear) feet leaving/landing together.
        self.rewards.pair_synchrony = mg.RewardTermCfg(
            func=v5_12_mdp.front_rear_pair_synchrony,
            weight=-2.00,
            params={
                "sensor_cfg": ordered_sensor,
                "command_name": "base_velocity",
                "moving_threshold": 0.05,
            },
        )

        # Cheat 2: one leg abandoned (the persistent rear-left problem).
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
