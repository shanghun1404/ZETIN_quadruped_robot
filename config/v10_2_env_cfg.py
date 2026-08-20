"""Stage 1 V10.2 -- re-apply the yaw band on the new robot, as one isolated change.

V9.2 established on the v1 robot that ``terrain_levels_vel`` scores straight-line
displacement (promote above ``size[0]/2`` = 4.0 m) while a sampled yaw rate held
for 10 s traces a circle of radius ``v/w``. At v = 0.6, w = 0.6 the maximum
displacement is 2.0 m, so promotion was arithmetically impossible; narrowing the
band to +/-0.2 moved the curriculum from a pinned 0.18 to a peak of 2.111.

The geometry is a property of the *curriculum and the command*, not of the robot,
so it should carry over. What does not carry over is the assumption inside it:
the chord formula uses the robot's actual speed, and the 45 kg robot's is not yet
known. V10.1 measures the baseline; V10.2 changes the yaw band and nothing else.

Everything else -- the robot, the rename, orientation termination, the V9.1-era
velocity kernel and stock gait reward -- is inherited from V10.1 unchanged.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from v10_1_env_cfg import GuideV2BaselineEnvCfg, V10_1_REWARD_WEIGHTS


V10_2_REWARD_WEIGHTS = dict(V10_1_REWARD_WEIGHTS)

# The V9.2 band. A 180 degree rotation about z leaves yaw rate unchanged, so this
# needs no guide-frame conversion.
V10_2_YAW_RATE_RANGE = (-0.2, 0.2)


@configclass
class GuideV2TurnBandEnvCfg(GuideV2BaselineEnvCfg):
    """V10.1 with a yaw band the terrain curriculum can reward."""

    def __post_init__(self):
        super().__post_init__()

        # The single isolated change from V10.1.
        self.commands.base_velocity.ranges.ang_vel_z = V10_2_YAW_RATE_RANGE
