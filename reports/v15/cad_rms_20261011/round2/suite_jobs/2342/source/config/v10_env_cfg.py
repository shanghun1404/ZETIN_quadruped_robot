"""Stage 1 V10 -- the V9.4 objective on the new 45 kg robot.

Why this is a new version rather than a config tweak
----------------------------------------------------
Every number in V8.1 through V9.4 was measured on the 25.34 kg v1 robot. The new
asset is 45.451 kg with a *lower* knee torque limit (48.0 N m against 53.0), so
its torque-to-weight ratio is roughly half. The duty floor at 0.35, the +/-0.2 yaw
band derived from a 4.0 m promotion chord, the 0.5 velocity kernel width and the
gait speed ramp are all carried over unchanged -- but they are hypotheses again,
not established results, and the first V10 run is what re-measures them.

What changes against V9.4, and why each is forced
--------------------------------------------------
1. **The robot.** ``GUIDE_ROBOT_V2_CFG``, with per-joint torque limits
   (HAA 24.8, HFE/KFE 48.0) instead of one 53.0 for all twelve.

2. **The trunk link is renamed** ``base_link`` -> ``base``. That name appears in
   three places in the inherited chain: the base-mass randomisation event, the
   height-scanner prim path, and the termination body list.

3. **Termination moves from contact to orientation.** V9.1 measured the right
   contact-termination body set on v1 (``base_link|.*_hip``, dropping thigh and
   calf because they legitimately touch stairs). That result cannot be applied
   here: the MJCF gives collision geometry to the four feet only, so ``base`` and
   the hips own no collider and ``illegal_contact`` could never fire -- a fallen
   robot would simply never terminate.

   Adding bounding-box colliders to those bodies was tried and measured to make
   things worse: placed from bounds that carry the conversion's baked offsets,
   the boxes hit the ground, tilted the robot and lifted two feet, dropping
   ground support from 99.4% to 56%.

   ``bad_orientation`` uses the projected-gravity tilt the policy already
   observes. It is a ratio, so it is immune to the frame/geometry offset this
   asset carries, and "the trunk has tipped past 60 degrees" is a more direct
   statement of falling than "something touched the floor" anyway.

4. **Spawn height.** CORRECTED IN V10.3. This previously read "at rest the root
   frame reads z = -0.419", and derived the spawn as ``clearance - 0.42``. Both
   signs were wrong: a zero-action trace reads the root at exactly the configured
   value before the first step and settles at +0.445, so the robot was spawned
   0.81 m underground for four versions. The spawn is now ``clearance +
   FEET_BELOW_ROOT_FRAME_M``. The per-body frame displacement this asset carries
   is real, but it does not apply to the articulation root.

The reward table is untouched -- the fifteen weights, the duty floor, the yaw
band, the velocity kernel and V9.4's speed-ramped gait all carry over exactly.
"""

from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import v10_mdp

from guide_robot_v2_cfg import (
    GUIDE_ROBOT_V2_CFG,
    GUIDE_V2_BASE_BODY,
    SPAWN_ROOT_FRAME_Z_M,
)
from v9_4_env_cfg import GuideTerrainTravelGatedEnvCfg, V9_4_REWARD_WEIGHTS


# The reward table is untouched; V10 is a robot change.
V10_REWARD_WEIGHTS = dict(V9_4_REWARD_WEIGHTS)

# Tilt past which the trunk is considered fallen. 60 degrees is well beyond any
# slope or stair the terrain generator produces, so it fires on falling rather
# than on climbing.
FALL_TILT_LIMIT_RAD = math.radians(60.0)

# Where the height scanner rides. v1 mounted it on ``base_link``.
SCANNER_PARENT_PATH = "{ENV_REGEX_NS}/Robot/base/base"
# Every rigid body of this asset lives one level deeper than the stock layout.
CONTACT_SENSOR_PATH = "{ENV_REGEX_NS}/Robot/base/.*_foot"


@configclass
class GuideRobotV2EnvCfg(GuideTerrainTravelGatedEnvCfg):
    """V9.4's objective, re-measured on the 45 kg robot."""

    def __post_init__(self):
        super().__post_init__()

        # 1. the robot itself
        self.scene.robot = GUIDE_ROBOT_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, SPAWN_ROOT_FRAME_Z_M)

        # 2. the trunk rename, everywhere the inherited chain names it
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=GUIDE_V2_BASE_BODY
        )
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = SCANNER_PARENT_PATH

        # The stock contact sensor path is "{ENV_REGEX_NS}/Robot/.*", which
        # IsaacLab resolves one path segment at a time. This asset nests every
        # rigid body one level deeper, at Robot/base/<body>, so the stock pattern
        # matches no body with a contact-report API and environment construction
        # fails. Only the feet carry colliders, so only the feet are watched.
        self.scene.contact_forces.prim_path = CONTACT_SENSOR_PATH

        # ``mdp.height_scan`` measures ground relative to the sensor's parent body
        # frame, and this asset's frames are displaced from their geometry by a
        # large constant, so no fixed ``offset`` corrects it. Calibrating the
        # offset to the settled frame value (-0.4192) was tried and MEASURED still
        # saturated: scan mean -0.969 with 96.9% of the 187 points pinned at the
        # -1 clip, and the matching run collapsed to terrain level 0.000 with a
        # mean episode length of 7.8 steps out of 1000. ``relative_height_scan``
        # returns median(hit_z) - hit_z, a difference between two ray hits, so the
        # body frame never enters it.
        scan_term = getattr(self.observations.policy, "height_scan", None)
        if scan_term is not None:
            scan_term.func = v10_mdp.relative_height_scan
            scan_term.params = {"sensor_cfg": SceneEntityCfg("height_scanner")}

        # 3. orientation-based fall detection, replacing contact termination
        self.terminations.body_contact = None
        self.terminations.base_tilt = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": FALL_TILT_LIMIT_RAD},
        )
