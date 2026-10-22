"""Stage 1 V9.2 -- a yaw command band the terrain curriculum can actually reward.

What V9.1 proved, and what it then ran into
-------------------------------------------
V9.1's termination fix worked completely::

                            V9 @4142      V9.1 @2270
    body_contact ends           82.5%          0.05%
    mean episode length      93 / 1000     1000 / 1000
    gait reward                  0.814          7.428
    base_linear_velocity         0.311          3.227

Those reward rates match what V8.1 achieved on *flat* ground (7.71 / 3.40), so
the robot walks on rough terrain about as well as it walks on a plane, and every
episode runs to full length. That part is finished.

But the terrain curriculum peaked early and then went backwards::

    iter  500 -> 1.584      iter 1500 -> 0.913
    iter 1000 -> 1.317      iter 2000 -> 0.716
                            iter 2270 -> 0.782

Out of ``num_rows = 10`` difficulty levels it sat below 1. The robot was walking
beautifully on the easiest terrain in the generator and being *demoted* off
anything harder.

Why: the curriculum measures displacement, the command produces circles
-----------------------------------------------------------------------
``terrain_levels_vel`` scores straight-line displacement from the spawn point::

    distance  = ||root_pos_w[:2] - env_origins[:2]||
    move_up   = distance > terrain_generator.size[0] / 2      # 8.0 / 2 = 4.0 m
    move_down = distance < ||cmd_xy|| * max_episode_length_s * 0.5   # cmd * 10 m

V8's command config, inherited unchanged through V9 and V9.1, samples a yaw rate
and holds it::

    heading_command=False,  ang_vel_z=(-0.6, 0.6),  resampling_time_range=(10, 10)

A constant speed ``v`` with a constant yaw rate ``w`` traces a circle of radius
``v/w``, so the displacement after time ``T`` is the chord::

    chord(v, w, T) = (2v / w) * |sin(w * T / 2)|

At ``v = 0.6``, ``w = 0.6`` that is a 1.0 m radius and a maximum displacement of
2.0 m -- against a 4.0 m promotion bar and a 6.0 m demotion bar. The robot could
walk perfectly for twenty seconds and still be demoted, every episode.

Stock IsaacLab never hits this because it sets ``heading_command=True`` with
``rel_heading_envs=1.0``: the yaw rate is *derived* from heading error, so the
robot turns once and then goes straight, and displacement grows with time. V8
turned that off, correctly, for a flat-ground guide task with no curriculum. The
incompatibility only exists once terrain and ``terrain_levels_vel`` are added.

The isolated change
-------------------
The yaw band narrows from +/-0.6 to +/-0.2 rad/s. Nothing else moves: the reward
table, the duty floor, the termination set, the scanner, the terrain generator,
the network and the PPO settings are all inherited from V9.1.

``heading_command`` stays **off** deliberately. Turning it on is the stock fix
and would guarantee promotion, but it removes the commanded yaw rate from the
policy's input distribution -- and a guide dog has to turn on command, because a
person decides where to go. Narrowing the band keeps that capability and buys
back the curriculum; switching to heading control would trade it away.

Why 0.2, from the chord formula rather than from taste
------------------------------------------------------
Worst case: a yaw rate held for the full 20 s episode with no sign change.

======  =========  =========  ============  =========
w       v = 0.3    v = 0.9    promote>4.0   verdict
======  =========  =========  ============  =========
0.6      0.84 m     2.52 m    no            never promotes
0.4      1.37 m     4.10 m    marginal      mostly demoted
**0.2**  **2.73 m** **8.18 m** yes at speed  promotes across most of the band
0.1      5.05 m    15.1 m     yes           promotes everywhere
======  =========  =========  ============  =========

At 0.2 the fast half of the speed band clears the 4.0 m bar outright, and because
``move_down`` is masked by ``~move_up`` a promoted environment is never demoted in
the same step. The slow-and-turning corner (``v = 0.3``, sustained ``w = 0.2``)
still demotes at 2.73 m; that corner is not eliminated, it is made a minority of
the distribution instead of nearly all of it. Real episodes do better than this
table because the command resamples at 10 s, so a sign change straightens the
path and increases the chord.

0.1 would promote everywhere and was not chosen: it is close enough to zero that
the policy would barely learn to turn at all, which is the capability the band
exists to preserve.

Falsifiable, and checked on a trend rather than a threshold
------------------------------------------------------------
V9.1's early check passed -- 0.9065 at iteration 1511, against a 0.5 bar -- and
the curriculum was *already declining* from a 1.584 peak at iteration 500. A
single threshold at a single iteration cannot see that. V9.2's check reads the
whole series and fails on a sustained decline from peak as well as on a low
absolute level, so the failure V9.1 hid is the failure it is built to catch.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from v9_1_env_cfg import GuideTerrainFixedEnvCfg, V9_1_REWARD_WEIGHTS


# The reward table is untouched; V9.2 is a command-range change.
V9_2_REWARD_WEIGHTS = dict(V9_1_REWARD_WEIGHTS)

# A 180 degree rotation about z leaves angular velocity about z unchanged, so
# this needs no guide-frame conversion (see v8_env_cfg).
V9_2_YAW_RATE_RANGE = (-0.2, 0.2)

# The curriculum's own constants, restated here so the tests can check the
# geometry rather than trusting the number.
TERRAIN_TILE_SIZE_M = 8.0
PROMOTION_DISTANCE_M = TERRAIN_TILE_SIZE_M / 2.0   # terrain_levels_vel: size[0] / 2
EPISODE_LENGTH_S = 20.0

# Curriculum health check. V9.1's single-threshold check passed while the level
# was falling, so V9.2 checks the trend too.
TERRAIN_LEVEL_CHECK_ITER = 1500
TERRAIN_LEVEL_MIN_EXPECTED = 1.5
TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK = 0.25


def displacement_chord_m(speed_m_s: float, yaw_rate_rad_s: float, seconds: float) -> float:
    """Straight-line displacement of a constant-speed, constant-yaw-rate path.

    This is exactly the quantity ``terrain_levels_vel`` compares against its
    promotion and demotion bars, so the command band can be chosen by
    calculation instead of by trial.
    """
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")
    if speed_m_s < 0.0:
        raise ValueError("speed must be non-negative")
    if yaw_rate_rad_s == 0.0:
        return speed_m_s * seconds

    import math

    radius = speed_m_s / abs(yaw_rate_rad_s)
    return 2.0 * radius * abs(math.sin(abs(yaw_rate_rad_s) * seconds / 2.0))


@configclass
class GuideTerrainTurnBandEnvCfg(GuideTerrainFixedEnvCfg):
    """V9.1's terrain task with a yaw band the curriculum can reward."""

    def __post_init__(self):
        super().__post_init__()

        # The single isolated change from V9.1.
        self.commands.base_velocity.ranges.ang_vel_z = V9_2_YAW_RATE_RANGE
