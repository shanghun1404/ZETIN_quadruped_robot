"""Stage 1 V10.1-flat -- command the robot along its actual fore-aft axis.

The measured error
------------------
V10-flat trained successfully on the new robot: 8/8 gait gates at iteration 2000,
0.503 m/s against a commanded 0.500, 100% survival, and a 49-53% duty split with
the duty floor never binding. But it was commanded along base **x**, and this
robot's fore-aft axis is **y**. It learned to travel sideways at the commanded
speed -- a crab walk, not a trot.

The axis was measured two independent ways on a standing robot::

    front_pair - rear_pair  = [0.181, 0.881, 0.191]   0.88 m apart in y
    left_pair  - right_pair = [-0.517, -0.245, -0.202] 0.52 m apart in x
    whole-robot extent xyz  = [0.80, 1.42, 0.58]       longest horizontal is y

The head points **+y**. The v1 robot's head pointed along base -x, which is why
the whole V8-V10 line converts a guide-forward speed to a negative base
``lin_vel_x``; that convention was inherited without rechecking it against the
new asset. The reference config for the old robot even carries the comment
"this robot's left/right axis is Y, and forward is +X" -- exactly inverted here.

Why this could not be caught by the metrics
--------------------------------------------
Every gate stayed valid. Forward speed was measured as ``-root_lin_vel_b[:, 0]``,
which is a real body-frame speed; the robot really did move at 0.503 m/s in the
commanded direction. Survival, duty factor and contact fractions are all
direction-agnostic. The only number that flinched was the FL-RR diagonal phase
correlation (-0.020 against a >= 0 bar), which a crab walk has no reason to
satisfy. The error was visible in the video before it was visible in the numbers.

The isolated change
-------------------
The forward command moves from base x to base +y::

    lin_vel_x : (-0.9, -0.3)  ->  (0.0, 0.0)
    lin_vel_y : (0.0, 0.0)    ->  (+0.3, +0.9)

Nothing else moves. The reward table, the duty floor, the termination set, the
robot, the action scale, the control rates, the network and the PPO settings are
all inherited from V10-flat. No lateral-velocity penalty is added, deliberately:
that would be a second simultaneous change, and the point of this run is to learn
whether the same objective produces a proper fore-aft trot once it is pointed the
right way.

Downstream
----------
``evaluate_v8`` and ``play_v8_video`` both assume guide-forward is base -x. They
take a ``--forward-axis`` selector so a V10.1 checkpoint is scored on ``+y`` and
filmed with the camera and head marker on the correct side. Scoring a V10.1
policy under the -x convention would report its forward speed as roughly zero.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from v10_flat_env_cfg import GuideV2FlatEnvCfg, V10_FLAT_REWARD_WEIGHTS


# The reward table is untouched; V10.1 is a command-direction change.
V10_1_FLAT_REWARD_WEIGHTS = dict(V10_FLAT_REWARD_WEIGHTS)

# The robot's head points along base +y; see the module docstring for the two
# independent measurements.
FORWARD_AXIS = "y"
FORWARD_SIGN = +1

# The same guide walking band V8 onward has used, now expressed on the axis the
# robot actually faces along.
V10_1_FORWARD_SPEED_RANGE = (0.3, 0.9)
V10_1_LATERAL_SPEED_RANGE = (0.0, 0.0)


@configclass
class GuideV2FlatForwardEnvCfg(GuideV2FlatEnvCfg):
    """V10-flat's objective, commanded along the robot's real fore-aft axis."""

    def __post_init__(self):
        super().__post_init__()

        # The single isolated change from V10-flat.
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = V10_1_LATERAL_SPEED_RANGE
        ranges.lin_vel_y = (
            FORWARD_SIGN * V10_1_FORWARD_SPEED_RANGE[0],
            FORWARD_SIGN * V10_1_FORWARD_SPEED_RANGE[1],
        )
