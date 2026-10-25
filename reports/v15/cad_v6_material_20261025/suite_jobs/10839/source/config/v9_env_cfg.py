"""Stage 1 V9 -- restore terrain sensing and rough ground on top of V8.1.

What V8 and V8.1 established
----------------------------
V8 ported the guide robot into the published IsaacLab Spot locomotion recipe and
produced, for the first time in this project, a textbook trot *phase*: both
same-side pairs anti-phase (FL-FR -0.486, RL-RR -0.229) and both diagonals in
phase (FL-RR +0.708, FR-RL +0.895) at the commanded speed. Every earlier version
decoupled exactly one pair and locked the other.

What V8 did not constrain was the duty factor: it converged to a 16/84 contact
split, one diagonal pair carrying the robot while the other stayed airborne.
V8.1 adds the single missing term -- a bounded floor under the least-used foot's
contact fraction -- and V9 builds on that table rather than on V8's, so terrain
is not being asked to fix a flat-ground defect.

V8 and V8.1 both run on **flat ground with no terrain sensor**: ``height_scanner``
is ``None`` and the policy observes 48 proprioceptive values only. That is fine
for a walking gait but it is not the guide-dog task, and Stage 2's vision plan is
defined as "a camera reproducing what the height scanner sees" -- so the scanner
has to exist before that distillation target exists.

The isolated change
-------------------
V9 adds terrain traversal. Two pieces are coupled and cannot be separated into
their own versions, because each is meaningless without the other:

* A **height scanner** is a constant zero vector on flat ground, so adding it
  alone would be a no-op the network learns to ignore.
* **Rough terrain** without a scanner is a different task -- blind traversal --
  not the one Stage 2 builds on.

Everything else is inherited from V8.1 unchanged: the full Spot reward table and
weights plus the duty floor, action scale 0.2, 50 Hz control, the MLP policy, the
guide-frame command direction, and the termination set. Only the observation
gains the scan and the ground gains relief.

Scanner geometry
----------------
The scanner is offset to **base x = -0.8 m**, which is *forward* in the guide
frame (the robot's head points along base -x; see ``v8_env_cfg``). With a
0.1 m grid over 1.6 x 1.0 m it returns 17 x 11 = 187 samples covering the ground
from under the robot to 1.6 m ahead. This is the same footprint the V5 line
used, chosen so a forward-facing camera can plausibly reproduce it in Stage 2.

Curriculum
----------
``max_init_terrain_level = 0`` starts every environment on the easiest terrain
and ``terrain_levels_vel`` promotes an environment only once it walks far enough
for its commanded speed. The robot therefore re-learns the V8.1 gait on near-flat
ground before it meets slopes and stairs, rather than being dropped onto rough
terrain with a randomly initialised policy.

Known risk
----------
The Spot table penalises body tilt (``base_orientation`` -3.0) and vertical
motion (``base_motion`` -2.0), both of which are unavoidable on slopes and
stairs. These weights are kept unchanged here so that any terrain failure is
attributable to the terrain itself rather than to a simultaneous reward edit.
If traversal stalls with the body pinned level, relaxing those two terms on
sloped terrain is the next isolated change, not a change to make now.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from v8_env_cfg import (
    V8ObservationsCfg,
    V8_BASE_LIN_VEL_X_RANGE,
    V8_GUIDE_FORWARD_SPEED_RANGE,
    V8_YAW_RATE_RANGE,
)
from v8_1_env_cfg import GuideDutyFloorEnvCfg, V8_1_REWARD_WEIGHTS


# The reward table is untouched; V9 is a perception and terrain change.
V9_REWARD_WEIGHTS = dict(V8_1_REWARD_WEIGHTS)

# Forward-looking scan footprint. -0.8 m in the base frame is 0.8 m *ahead* of
# the robot, because the guide frame points along base -x.
SCAN_OFFSET_X = -0.8
SCAN_RESOLUTION = 0.1
SCAN_SIZE = [1.6, 1.0]
SCAN_ROWS, SCAN_COLS = 17, 11
SCAN_POINTS = SCAN_ROWS * SCAN_COLS  # 187


# ---------------------------------------------------------------------------
# Dedicated stair-climbing evaluation terrain
# ---------------------------------------------------------------------------
#
# Training terrain (ROUGH_TERRAINS_CFG) already contains 20% ascending and 20%
# descending stairs, but an environment lands on a random tile, so a rough-terrain
# evaluation cannot say anything specific about stair climbing. This terrain is
# stairs only, so the stair result is measured rather than inferred.
#
# Ascending vs descending: the terrain origin -- where the robot spawns -- sits at
# ``+(num_steps + 1) * step_height`` for a pyramid and at ``-(num_steps + 1) *
# step_height`` for an inverted pyramid. So a pyramid spawns the robot on the top
# platform and walking outward goes *down*, while an inverted pyramid spawns it in
# the pit and walking outward goes *up*. Climbing therefore needs the inverted
# variant.
#
# ``num_rows`` is the difficulty axis: with ``curriculum=True`` row 0 gets the
# 0.05 m step and the last row the 0.20 m step, so one evaluation reports how
# high a step the policy can actually clear. 0.20 m brackets a real building
# stair (0.15-0.18 m), which is the case that matters for a guide dog.
STAIRS_STEP_HEIGHT_RANGE = (0.05, 0.20)
STAIRS_STEP_WIDTH = 0.30
STAIRS_NUM_LEVELS = 8

STAIRS_UP_EVAL_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=STAIRS_NUM_LEVELS,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=STAIRS_STEP_HEIGHT_RANGE,
            step_width=STAIRS_STEP_WIDTH,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

STAIRS_DOWN_EVAL_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=STAIRS_NUM_LEVELS,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=STAIRS_STEP_HEIGHT_RANGE,
            step_width=STAIRS_STEP_WIDTH,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class V9ObservationsCfg(V8ObservationsCfg):
    @configclass
    class PolicyCfg(V8ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class V9CurriculumCfg:
    """Promote an environment's terrain level once it walks far enough."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class GuideTerrainEnvCfg(GuideDutyFloorEnvCfg):
    """V8.1's duty-floored trot, now with terrain sensing on rough ground."""

    observations: V9ObservationsCfg = V9ObservationsCfg()
    curriculum: V9CurriculumCfg = V9CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # V8's ``__post_init__`` sets ``self.curriculum = None`` for flat ground,
        # and it runs *after* the class attribute above is bound -- so the
        # declaration alone leaves V9 with no terrain promotion at all, pinned to
        # the easiest tile for the entire run. Restore it here, after the parent
        # has had its say. (Caught by probe_v9: ``curriculum_terms`` came back
        # empty while the source clearly declared the term.)
        self.curriculum = V9CurriculumCfg()

        # V8 disabled both of these for flat ground; V9 turns them back on.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        # Start everyone on the easiest tile and let the curriculum promote.
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(SCAN_OFFSET_X, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=SCAN_RESOLUTION, size=SCAN_SIZE
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
