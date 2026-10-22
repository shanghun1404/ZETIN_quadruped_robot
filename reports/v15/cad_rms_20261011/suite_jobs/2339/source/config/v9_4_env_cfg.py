"""Stage 1 V9.4 -- stop paying the gait reward to a robot that is not travelling.

What V9.3 settled
-----------------
Sharpening the linear-velocity kernel (std 1.0 -> 0.5) worked, and was not
enough. Measured against V9.2 at matched iterations::

    iteration          2000    2500    3000    4000    ~5000
    V9.2 terrain lvl  1.417   0.797   0.401     --     0.265
    V9.3 terrain lvl  1.411   1.359   1.393   1.190    0.864
    V9.2 speed m/s    0.156     --    0.085   0.091    0.085
    V9.3 speed m/s    0.224     --    0.219   0.209    0.158

At iteration 3000 the curriculum is 3.5x higher and the speed 2.6x higher, so
the kernel was a real constraint. But the robot still travels a fifth of what it
is told to, still climbs no stairs, and the curriculum still decays.

Why: the rest of the table pays for standing still
--------------------------------------------------
V9.3's reward rates at iteration 4999, against what a *stationary* robot scores::

    term                     measured   stationary
    base_angular_velocity      4.543      4.524     yaw cmd 0.2, std 2.0
    gait                       6.745      ~7        contact timing only
    air_time                   0.872      ~0.9
    base_linear_velocity       2.226      1.506     cmd 0.6, std 0.5
                             -------    -------
    positive total            14.690      ~13.6     = 93%

Standing still and cycling the legs collects 93% of the available reward.
Travelling perfectly would add about 3.5 per second on the velocity term while
paying more foot slip, more action smoothness, more base motion, and the risk of
falling. Declining is rational.

Two terms are responsible. ``base_angular_velocity`` became nearly free when V9.2
narrowed the yaw band to +/-0.2 -- a still robot tracks a near-zero yaw command
almost exactly, and 4.52 of 5.00 is collected for not rotating. And ``GaitReward``
gates on ``cmd > 0.0 OR body_vel > threshold``, so being *commanded* to move is
enough to unlock all 10.0 of it.

The isolated change
-------------------
The gait reward is scaled by actual travel speed:
``ramp = clamp(body_speed / velocity_threshold, 0, 1)``. Nothing else moves --
the fifteen weights, the kernel widths, the yaw band, the termination set, the
duty floor, the scanner, the terrain generator, the network and the PPO settings
are all inherited from V9.3.

``base_angular_velocity`` is deliberately left alone even though it is the second
largest free term, because two simultaneous edits would make the result
unattributable. It is the next lever if this one is not enough.

Sizing, at the measured behaviour
---------------------------------
At V9.3's measured 0.22 m/s with ``velocity_threshold = 0.5`` the ramp is 0.44,
so the gait reward falls from 6.745 to about 2.97 -- a loss of **3.8 per second**,
three times what V9.3's kernel change was worth, and the largest single term in
the table. Every increment of speed recovers part of it, and at the commanded
0.5 m/s the ramp saturates and the published recipe is scored unchanged.

Falsifiable
-----------
If forward speed is still under 0.3 m/s by iteration 2500, then removing the
gait reward's contribution to standing was also not sufficient, and the next
lever is ``base_angular_velocity`` -- either a narrower ``std`` than 2.0 or the
same speed ramp applied to it.
"""

from __future__ import annotations

from isaaclab.utils import configclass

import v9_4_mdp
from v9_3_env_cfg import GuideTerrainSharpVelEnvCfg, V9_3_REWARD_WEIGHTS


# The weight table is untouched; V9.4 changes when the gait reward is earned.
V9_4_REWARD_WEIGHTS = dict(V9_3_REWARD_WEIGHTS)

# Speed at which the gait reward is scored in full, in m/s. This is the Spot
# recipe's own ``velocity_threshold`` for the term, reused rather than invented
# so the saturation point stays where the published config put it.
GAIT_FULL_CREDIT_SPEED_M_S = 0.5

# V9.3's measured forward speed, used in the docstring arithmetic above and by
# the tests that check this change is worth more than the last one.
V9_3_MEASURED_SPEED_M_S = 0.22


@configclass
class GuideTerrainTravelGatedEnvCfg(GuideTerrainSharpVelEnvCfg):
    """V9.3's terrain task with the gait reward scaled by actual travel."""

    def __post_init__(self):
        super().__post_init__()

        # The single isolated change from V9.3: same term, same weight, same
        # parameters -- only the class, which scales the stock reward by speed.
        self.rewards.gait.func = v9_4_mdp.SpeedRampedGaitReward
