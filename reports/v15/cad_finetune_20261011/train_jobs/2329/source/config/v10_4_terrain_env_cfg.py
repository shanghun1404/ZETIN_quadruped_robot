"""Stage 1 V10.4 -- V10.3's gait on generated rough terrain, blind.

What this run is
----------------
V10.3 is the first configuration on this robot that passes all eight gait gates
with an honest fore-aft trot: contact fractions within 1% of even on all four
feet, `FR-RL` +0.7124, 64/64 survival, and a training run with a 0.0012 fall
rate. It has never left flat ground. This takes that policy's objective onto
generated terrain **without a height scanner** -- blind traversal, proprioception
only.

Blind is the deliberate choice, not a simplification
----------------------------------------------------
Every scanner-based attempt on this asset failed for reasons that traced back to
the scanner, not the gait. ``mdp.height_scan`` measures ground relative to the
*sensor's parent body frame*, and this conversion leaves body frames displaced
from their geometry, so the scan saturated: 96.9% of 187 points pinned at the
-1 clip, and the matching run collapsed to terrain level 0.000 with a mean
episode length of 7.8 steps out of 1000. ``v10_mdp.relative_height_scan`` fixed
the saturation by differencing two ray hits, but that repair was never separated
from the other terrain failures.

Removing the scanner removes that entire class of coupling. It also keeps the
observation at V8's 48 dimensions, which means ``evaluate_v8.py`` and
``play_v8_video.py`` score and film a V10.4 checkpoint with no changes, and a
V10.3 checkpoint and a V10.4 checkpoint are directly comparable.

What changes from V10.3, and why each is forced
-----------------------------------------------
1. **Terrain generation on.** ``terrain_type = "generator"`` with
   ``ROUGH_TERRAINS_CFG`` -- the same terrain V9 used, so a V10.4 result is
   comparable with that line. ``max_init_terrain_level = 0`` starts every
   environment on the easiest tile.

2. **The curriculum, restored after ``super()``.** ``terrain_levels_vel``
   promotes an environment once it walks past half a tile. V8's
   ``__post_init__`` sets ``self.curriculum = None`` for flat ground and it runs
   *after* a class attribute is bound, so declaring the curriculum as a class
   attribute alone silently leaves the run with no promotion at all. That
   already cost one version (V9, caught by its probe before launch) and is
   re-asserted by this version's probe.

3. **The yaw band narrows to +/-0.2.** This is a precondition for the curriculum
   to function, not a taste change. ``terrain_levels_vel`` promotes when the
   straight-line displacement from the tile origin exceeds ``size[0] / 2`` =
   4.0 m. A robot holding speed ``v`` and yaw rate ``w`` for ``T`` seconds
   traces a circular arc whose chord is ``(2v/w)·|sin(wT/2)|``::

       yaw +/-0.6 (inherited from V8)   chord 0.466 m   promotion impossible
       yaw +/-0.2 (established in V9.2) chord 4.546 m   promotion possible

   At the inherited band no environment could ever be promoted regardless of how
   well it walked, and the run would be wasted. V9.2 measured this and the
   curriculum climbed to 2.111 under it.

The scanner stays ``None``, the reward table is untouched, and the spawn height,
termination set, action scale, control rates, network and PPO settings all carry
over from V10.3 exactly.

The spawn, which is the open risk
---------------------------------
V10.3 corrected a sign-inverted spawn constant that had been putting this robot
0.81 m underground. On flat ground the corrected value is +0.495 and the probe
measures zero contact force on the first step. The terrain value is +0.765
(0.32 m clearance + 0.445 m foot-to-root) and **has never been run**.

It may not be enough. On terrain a robot spawns at ``env_origin +
init_state.pos`` and each tile's origin sits at its own local surface, so a
raised tile can put its surface above the feet even at this height. The
predecessor report records a +0.6 m frame spawn that made contact and still
flipped to 165.6 degrees. ``probe_v10_4_terrain.py`` measures the spawn state
relative to the environment origin and fails the run before training if the
robot starts embedded.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from v10_3_flat_env_cfg import (
    FORWARD_AXIS,
    FORWARD_SIGN,
    GuideV2FlatSpawnFixedEnvCfg,
    V10_3_FLAT_REWARD_WEIGHTS,
)
from guide_robot_v2_cfg import FEET_BELOW_ROOT_FRAME_M, FEET_SPAWN_CLEARANCE_M


# The reward table is untouched; V10.4 is a terrain change.
V10_4_REWARD_WEIGHTS = dict(V10_3_FLAT_REWARD_WEIGHTS)

# Narrowed from V8's inherited (-0.6, 0.6). See the docstring for the chord
# arithmetic that makes this a precondition rather than a preference.
V10_4_YAW_RATE_RANGE = (-0.2, 0.2)

# ``terrain_levels_vel`` promotes on ``terrain_generator.size[0] / 2``.
TERRAIN_TILE_SIZE_M = 8.0
PROMOTION_DISTANCE_M = TERRAIN_TILE_SIZE_M / 2.0

# Spawn on terrain: the root frame this far above the tile origin, which puts the
# feet ``FEET_SPAWN_CLEARANCE_M`` above it.
TERRAIN_SPAWN_ROOT_FRAME_Z_M = FEET_SPAWN_CLEARANCE_M + FEET_BELOW_ROOT_FRAME_M


def displacement_chord_m(speed_m_s: float, yaw_rate_rad_s: float, seconds: float) -> float:
    """Straight-line distance covered while turning at a constant rate.

    The quantity ``terrain_levels_vel`` compares against half a tile. A robot
    that turns fast enough traces a small circle and never gets far from its
    origin no matter how well it walks.
    """
    import math

    if yaw_rate_rad_s == 0.0:
        return speed_m_s * seconds
    return abs((2.0 * speed_m_s / yaw_rate_rad_s) * math.sin(yaw_rate_rad_s * seconds / 2.0))


@configclass
class V10_4CurriculumCfg:
    """Promote an environment's terrain level once it walks far enough."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class GuideV2BlindTerrainEnvCfg(GuideV2FlatSpawnFixedEnvCfg):
    """V10.3's trot on rough terrain, proprioception only."""

    curriculum: V10_4CurriculumCfg = V10_4CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # V8's __post_init__ sets self.curriculum = None and runs after the class
        # attribute above is bound. Without this line the run has no promotion at
        # all and sits on the easiest tile forever. This exact trap cost V9.
        self.curriculum = V10_4CurriculumCfg()

        # Terrain back on; the scanner deliberately stays off.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        self.scene.height_scanner = None

        # Spawn relative to the tile origin, with the corrected sign.
        self.scene.robot.init_state.pos = (0.0, 0.0, TERRAIN_SPAWN_ROOT_FRAME_Z_M)

        # Without this the curriculum cannot promote; see the docstring.
        self.commands.base_velocity.ranges.ang_vel_z = V10_4_YAW_RATE_RANGE
