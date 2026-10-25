"""Stage 1 V9.3 -- make standing still stop paying for itself.

What V9.2 settled, and what it exposed
--------------------------------------
V9.2's yaw band worked exactly as calculated. The terrain curriculum, pinned at
0.18 for all of V9 and declining through all of V9.1, climbed::

    iter    0 -> 0.987      iter 1479 -> 2.111  (peak)
    iter  500 -> 1.583      iter 1501 -> 2.103  (early check: PASS)

Then it collapsed over the remaining 3500 iterations::

    2000 -> 1.417   2500 -> 0.797   3000 -> 0.401   4999 -> 0.265

Evaluation says why. Across every sampled checkpoint the robot **does not
travel**::

    iteration      2000     3000     4000     4999
    speed m/s     0.156    0.085    0.091    0.085     (commanded 0.500)
    min contact   0.678    0.690    0.643    0.664
    world z m     0.282    0.259    0.276    0.283     (old evaluator output)
    survival      1.000    1.000    1.000    1.000
    stairs climb  0.000    0.000    0.000    0.000

The historical ``world z`` row is not comparable with the 0.3616 m calibrated
clearance on generated terrain: terrain origins carry their own z offsets. The
evaluator now subtracts the nearest finite height-scanner hit before reporting
``mean_base_height_m``. The speed and zero-travel diagnosis above is unchanged.

The stairs recording shows it plainly: at frames 20, 200 and 400 -- eight seconds
apart -- the robot is still in the pit at the foot of the stairs, splayed low,
never touching a step. It survives every episode by not attempting anything.

Why the reward table permits it
-------------------------------
``base_linear_velocity_reward`` is an exponential kernel on tracking error::

    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple

At ``std = 1.0`` a **stationary** robot under a 0.6 m/s command still scores
``exp(-0.6) = 0.549``, which at weight 5.0 is **2.74 per second**. The measured
value at iteration 4999 was 2.99. Tracking the command perfectly would score
5.00, so the entire incentive to move at all is about 2.0 per second -- and on
rough terrain that is cheaper to decline than to earn.

The two terms that look like they should force motion do not:

* ``GaitReward`` scores contact *timing* between foot pairs. A robot cycling its
  legs on the spot satisfies all six comparisons. V9.2 collected 6.44 of 10 with
  zero displacement.
* ``air_time_reward`` is gated on ``cmd > 0.0 or body_vel > velocity_threshold``,
  so merely being *commanded* to move unlocks it; it then pays for air and
  contact durations, never for distance.

Nothing in the fifteen terms mentions displacement. This is the same structural
error as V8's: the objective describes a *pattern* and omits the *quantity*. There
it was duty factor, here it is travel.

The isolated change
-------------------
``base_linear_velocity`` keeps its 5.0 weight and its function; only the kernel
width narrows, ``std`` 1.0 -> 0.5. Nothing else moves -- the other fourteen
weights, the duty floor, the termination set, the yaw band, the scanner, the
terrain generator, the network and the PPO settings are all inherited from V9.2.

Why 0.5, on two criteria at once
--------------------------------
Let ``e`` be the tracking error. Standing still under command ``c`` gives error
``c``, so the *payoff for moving* is ``5.0 * (1 - exp(-c/std))`` and the local
*gradient* the policy feels at that error is ``(5.0/std) * exp(-c/std)``.

At the measured operating point ``c = 0.6``:

std     | standing pays | moving is worth | weighted gradient at e=0.6
1.00    | 2.74/s        | 2.26/s          | 2.744
**0.50**| **1.50/s**    | **3.50/s**      | **3.012**
0.35    | 0.90/s        | 4.10/s          | 2.572
0.25    | 0.46/s        | 4.55/s          | 1.814

0.5 is the only value in that range that improves *both*: it cuts the standing
payoff by 45% and raises the gradient the policy actually experiences. Sharper
kernels keep improving the payoff gap but flatten the reward far from target,
which is the classic way a narrow exponential stops teaching anything to a policy
that starts out stationary -- precisely the state V9.2 is stuck in.

Falsifiable, and watched continuously this time
-----------------------------------------------
V9.2's early check passed at iteration 1501 and was right to: the curriculum was
at 2.10 and climbing. The collapse happened afterwards, and nothing looked again
for three and a half hours. So the check no longer stops at its first verdict --
it runs for the whole job and raises an alert the moment the level falls away
from its peak. On V9.2's own series that fires at about iteration 2500, where the
level had dropped 62% from 2.111 to 0.797.

The terrain level is the right thing to watch, and the obvious alternative is
not. ``Metrics/base_velocity/error_vel_xy`` looked like a direct speed proxy
until it was checked against runs with known outcomes::

    V8.1 flat, walking at 0.599 m/s     0.394
    V9.1 terrain, walking               1.073
    V9.2 terrain, stationary            1.151

V9.1 and V9.2 are 0.08 apart, so that metric cannot tell walking from standing on
terrain -- it is a mean over environments and a minority of tumbling ones swamp
it. ``terrain_levels`` is computed from displacement by definition, which is
exactly the quantity in question.

If the level still collapses after the peak, the kernel width was not the binding
constraint and the next lever is an explicit displacement reward rather than a
sharper velocity kernel.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from v9_2_env_cfg import GuideTerrainTurnBandEnvCfg, V9_2_REWARD_WEIGHTS


# The weight table is untouched; V9.3 narrows one kernel width.
V9_3_REWARD_WEIGHTS = dict(V9_2_REWARD_WEIGHTS)

# Exponential kernel width for base_linear_velocity_reward. See the module
# docstring for the payoff/gradient table this is chosen from.
V9_3_LIN_VEL_STD = 0.5
V9_2_LIN_VEL_STD = 1.0

# The command band the arithmetic above is evaluated at, in guide-frame speeds.
OPERATING_SPEED_M_S = 0.6


def standing_still_reward_per_s(command_m_s: float, std: float, weight: float = 5.0) -> float:
    """What a completely stationary robot collects from the velocity term.

    This is the quantity V9.3 exists to reduce, so it is computable rather than
    asserted: ``exp(-error/std)`` with ``error == command`` when velocity is zero.
    """
    import math

    if std <= 0.0:
        raise ValueError("std must be positive")
    return weight * math.exp(-abs(command_m_s) / std)


@configclass
class GuideTerrainSharpVelEnvCfg(GuideTerrainTurnBandEnvCfg):
    """V9.2's terrain task with a velocity kernel that discriminates."""

    def __post_init__(self):
        super().__post_init__()

        # The single isolated change from V9.2.
        self.rewards.base_linear_velocity.params["std"] = V9_3_LIN_VEL_STD
