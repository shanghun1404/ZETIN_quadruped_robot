"""Stage 1 V8.1 -- put a floor under the duty factor the Spot recipe leaves free.

What V8 fixed, and what it did not
----------------------------------
V8 solved gait *phase*. For the first time in this project all four correlation
gates passed at once (8000 iterations, ``model_7999``, commanded 0.5 m/s)::

    FL-FR  -0.486   same-side front, anti-phase   PASS
    RL-RR  -0.229   same-side rear,  anti-phase   PASS
    FL-RR  +0.708   diagonal,        in phase     PASS
    FR-RL  +0.895   diagonal,        in phase     PASS

Every version from V5.13 to V7 decoupled exactly one pair and locked the other.
The +10.0 gait reward ended that, and it is kept here untouched.

But the *duty factor* collapsed::

    FL 0.161   FR 0.789   RL 0.799   RR 0.162    contact fraction

One diagonal pair is airborne 84% of the time while the other carries the robot.
The phase pattern is a textbook trot; the weight distribution is a two-legged
drag. Base height fell to 0.302 m (0.347 m at iteration 1600, calibrated stance
0.3616 m) and survival to 0.922. V8 at 8000 iterations is visibly worse than V8
at 1600 -- it converged *into* this solution rather than away from it.

Why the Spot table permits this
-------------------------------
Two terms look like they should forbid it and neither does:

* ``GaitReward`` scores only **relative** timing. ``_sync_reward_func`` asks that
  the two feet of a diagonal pair share an air time and share a contact time;
  ``_async_reward_func`` asks that one pair's air time match the other pair's
  contact time. A 50/50 split satisfies every one of those six comparisons. So
  does 84/16. The reward is exactly indifferent between them.
* ``air_time_reward`` scores ``min(max(air_time, contact_time), mode_time)`` --
  the *larger* of the two. It pays the same for a foot that stays up a long time
  as for a foot that stays down a long time, so it rewards **both** extremes and
  penalizes only the balanced middle relative to them.

Nothing else in the fourteen terms mentions how long a foot must be loaded.
The 84/16 split is therefore not a training artifact to wait out; it is an
optimum of the objective as written.

The isolated change
-------------------
One term is added and nothing else moves: a bounded penalty on the least-used
foot falling below a contact-duty floor. The implementation is
``v5_12_mdp.abandoned_foot_penalty``, which is not new code -- it is the one
V5.12 penalty that demonstrably moved its own metric, and it is reused verbatim
so V8.1 inherits a term with a measured track record rather than a fresh guess.

Floor 0.35: below the ~0.5 duty factor of a healthy trot at 0.5 m/s, so it costs
a good gait exactly nothing, and above the 0.30 the evaluator gates on, so the
policy is trained with margin against the number it is judged by. The penalty
is bounded in [0, 1] and uses the *worst* foot rather than the mean, matching
``min_foot_contact_fraction`` -- the quantity trained against is the quantity
reported.

Sizing the weight against V8's own measured reward rates
--------------------------------------------------------
V8's TensorBoard at iteration 7999 (per-second rates, episodic sum / 20 s)::

    gait                    +7.565      action_smoothness   -1.267
    base_angular_velocity   +3.859      joint_acc           -0.869
    base_linear_velocity    +3.395      joint_pos           -0.692
    air_time                +1.889      base_motion         -0.436
    foot_clearance          +0.389      foot_slip           -0.240
                           -------      joint_vel           -0.218
    positive total          +17.10      base_orientation    -0.199
                                        joint_torques       -0.049
                                        air_time_variance   -0.021
                                                           -------
                                        negative total       -3.99

At the measured 0.161 duty the deficit is ``(0.35 - 0.161) / 0.35 = 0.540``, so
weight -3.0 produces **1.62 per second** -- larger than any existing penalty in
the table, and 9.5% of the positive total. It cannot override the gait reward it
is meant to complement (bounded worst case 3.0 against gait's 7.6), and once
every foot clears the floor it is identically zero, so a converged policy pays
nothing for carrying it.

For scale, V6 established what an *inert* term looks like on this robot:
``base_height_l2`` produced 0.022 per step against a 5.0 reward, and tripling
its weight moved the measured height 1.4 mm. This term starts 74x larger
relative to the reward it competes with.

Falsifiable
-----------
If ``min_foot_contact_fraction`` still sits near 0.16 after training, the duty
factor is not reachable by penalizing it and the next lever is
``air_time_reward`` -- replacing ``max(air, contact)`` with a term that pays for
the balanced middle instead of both extremes. That is a change to a maintained
upstream function and is deliberately *not* made here, so that any movement in
duty factor is attributable to this penalty alone.

Everything else -- the fourteen Spot weights, action scale 0.2, 50 Hz control on
200 Hz physics, the MLP policy, the guide-frame command direction, flat ground,
the termination set -- is inherited from V8 unchanged. The observation and
action spaces are byte-identical to V8's, which is what makes ``evaluate_v8.py``
and ``play_v8_video.py`` valid on a V8.1 checkpoint without a fork.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import v5_12_mdp
from v8_env_cfg import GuideSpotRecipeEnvCfg, ORDERED_FEET, V8_REWARD_WEIGHTS


# Minimum fraction of time each foot must spend loaded while the robot is
# commanded to move. Below a healthy trot's ~0.5, above the evaluator's 0.30.
DUTY_CONTACT_FLOOR = 0.35

# See the module docstring for how this is sized against V8's measured reward
# rates: 1.62/s at the failure, 0.0 once the floor is met, bounded at 3.0.
DUTY_FLOOR_WEIGHT = -3.00

# Contact force threshold, in newtons, for calling a foot "loaded". Matches the
# 5.0 N the evaluator uses, so training and reporting agree on what contact is.
DUTY_CONTACT_THRESHOLD = 5.0

# Averaging window for the per-foot duty estimate. One second is 50 control
# steps and 2-3 strides at the commanded speed, long enough to describe a duty
# factor and short enough to react inside a 20 s episode.
DUTY_WINDOW_S = 1.0

V8_1_REWARD_WEIGHTS = {
    **V8_REWARD_WEIGHTS,
    "duty_floor": DUTY_FLOOR_WEIGHT,
}


@configclass
class GuideDutyFloorEnvCfg(GuideSpotRecipeEnvCfg):
    """V8's Spot recipe with a contact-duty floor under the gait it produces."""

    def __post_init__(self):
        super().__post_init__()

        # ``preserve_order`` keeps the sensor columns in FL, FR, RL, RR order so
        # the reported per-foot contact fractions line up with the evaluator's.
        # The penalty itself takes a max over the four and is order-agnostic.
        ordered_feet_sensor = SceneEntityCfg(
            "contact_forces", body_names=list(ORDERED_FEET), preserve_order=True
        )

        # The single isolated change from V8.
        self.rewards.duty_floor = RewardTermCfg(
            func=v5_12_mdp.abandoned_foot_penalty,
            weight=DUTY_FLOOR_WEIGHT,
            params={
                "threshold": DUTY_CONTACT_THRESHOLD,
                "sensor_cfg": ordered_feet_sensor,
                "command_name": "base_velocity",
                "moving_threshold": 0.05,
                "window_s": DUTY_WINDOW_S,
                "contact_floor": DUTY_CONTACT_FLOOR,
            },
        )
