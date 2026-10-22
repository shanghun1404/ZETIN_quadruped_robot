"""Stage 1 V6: V5.13 anti-cheat rewards with a posture that actually stands up.

Evidence this is the right isolated change
------------------------------------------
The robot has been measured symmetric to within 0.68 mm (base-frame diagnostic,
all randomization disabled), so the front-pair coupling is not structural. Two
facts point at posture instead:

1. **The continuous in-phase penalty works.** V5.13 drove the rear pair from
   +0.349 to -0.037 with the exact same term that left the front pair at +0.685.
   The mechanism is sound; something else holds the front pair together.
2. **The robot crouches, consistently.** V5.11, V5.12 and V5.13 all settle near
   0.34 m against a 0.45 m target -- 10 cm low, three versions running. In a
   crouch the front legs have little usable travel and the forward-shifted load
   is cheapest to carry on both front legs at once, which is exactly the
   coupling that survives the penalty.

``base_height_l2`` was -2.00 against a forward-tracking reward of +5.00, so the
policy was rationally trading posture for speed. V6 raises it to -6.00, making
the crouch more expensive than the speed it buys.

Everything else -- reward table, penalties, network, robot, command speed --
is inherited from V5.13 unchanged, so any change in the front-pair correlation
is attributable to the posture weight alone.

Falsifiable: if the robot reaches ~0.45 m and the front pair stays coupled, the
crouch hypothesis is dead and the cause is elsewhere.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from v5_13_env_cfg import MeasuredAntiCheatGuideEnvCfg, V5_13_REWARD_WEIGHTS


BASE_HEIGHT_WEIGHT = -6.00

V6_REWARD_WEIGHTS = {
    **V5_13_REWARD_WEIGHTS,
    "base_height_l2": BASE_HEIGHT_WEIGHT,
}


@configclass
class UprightAntiCheatGuideEnvCfg(MeasuredAntiCheatGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # The single isolated change from V5.13.
        self.rewards.base_height_l2.weight = -6.00
