"""Stage 1 V10.1 -- the V9.1-equivalent baseline, re-measured on the 45 kg robot.

Why deliberately revert three fixes
------------------------------------
V10 carries the whole V9.1 -> V9.4 fix stack forward. Every one of those fixes was
derived from numbers measured on the 25.34 kg v1 robot, and the new robot has
roughly half its torque-to-weight ratio:

* the +/-0.2 yaw band came from a chord calculation against a 4.0 m promotion
  distance, assuming the robot reaches its commanded speed;
* the 0.5 velocity-kernel width came from what a *stationary* v1 robot collected;
* the gait speed ramp came from v1's measured 0.22 m/s.

None of that is established on this robot. Applying four unvalidated changes at
once and getting a bad result would leave nothing attributable -- the same trap
the V5.6-V7 line fell into.

So V10.1 reverts to the V9.1-era settings and becomes the baseline this robot's
fixes are measured against. V10.2 then re-applies the yaw band as a single
isolated change, exactly as V9.2 did on v1.

What is kept from V10 (not revertible, and not experiment variables)
--------------------------------------------------------------------
* the robot asset itself, with per-joint torque limits;
* the ``base_link`` -> ``base`` rename through the whole config chain;
* orientation-based termination -- contact termination is *impossible* here, not
  merely worse: the MJCF gives collision geometry to the four feet only, so
  ``base`` and the hips own no collider and ``illegal_contact`` could never fire.

What is reverted
----------------
======================  =============  =============
setting                 V10 (from v1)  V10.1
======================  =============  =============
yaw command band        +/-0.2         **+/-0.6**
velocity kernel ``std`` 0.5            **1.0**
gait reward             speed-ramped   **stock Spot**
======================  =============  =============

The reward *weights* are untouched throughout; these are kernel and command
settings, not weights.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp

from v10_env_cfg import GuideRobotV2EnvCfg, V10_REWARD_WEIGHTS


V10_1_REWARD_WEIGHTS = dict(V10_REWARD_WEIGHTS)

# V8/V9/V9.1 values, restored as this robot's baseline.
V10_1_YAW_RATE_RANGE = (-0.6, 0.6)
V10_1_LIN_VEL_STD = 1.0


@configclass
class GuideV2BaselineEnvCfg(GuideRobotV2EnvCfg):
    """New robot, V9.1-era objective. The baseline, not a candidate."""

    def __post_init__(self):
        super().__post_init__()

        # Revert V9.2's yaw band.
        self.commands.base_velocity.ranges.ang_vel_z = V10_1_YAW_RATE_RANGE
        # Revert V9.3's kernel width.
        self.rewards.base_linear_velocity.params["std"] = V10_1_LIN_VEL_STD
        # Revert V9.4's speed ramp, back to the published Spot gait reward.
        self.rewards.gait.func = spot_mdp.GaitReward
