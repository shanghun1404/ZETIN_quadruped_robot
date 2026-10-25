"""Stage 1 V10-flat -- the V8.1 objective on the new 45 kg robot, flat ground.

Why flat first
--------------
The new asset works perfectly on a flat ground plane and fails only on generated
terrain. Measured on flat, dropped and left to settle::

    per-foot vertical force   108.4 / 110.1 / 116.3 / 108.4 N
    total                     443.2 N  against 445.9 N own weight  = 99.4%
    tilt                      upright, stable

On generated terrain the same robot either falls through the mesh (spawn frame z
-0.10, zero contact force, root z reaching -314 m) or lands and flips to 165-166
degrees from every spawn height that does make contact. Three separate mechanisms
had already been broken by the same root cause -- the conversion leaves body
frames displaced from their geometry, so ``body_pos_w`` and ``mdp.height_scan``
mean something other than what they appear to mean.

**Corrected in V10.3:** ``init_state.pos`` was wrongly included in that list. A
zero-action trace reads the root at exactly the configured value before the first
step, so ``init_state.pos.z`` IS ``root_pos_w[:, 2]`` for the articulation root.
Applying the per-body frame argument to it produced a sign-inverted spawn constant
that put this robot 0.81 m underground for four consecutive versions.

Flat ground removes every one of those couplings: no terrain origins, no height
scanner, no curriculum. What it still measures is the thing that actually has to
work first -- whether this robot can walk at all, at half the torque-to-weight
ratio of the one every reward number was tuned on.

The objective
-------------
V8.1's, unchanged: the published Spot table plus the duty floor, which is the
configuration that passed all eight gait gates on the v1 robot. Terrain, the
scanner and the curriculum are already off in that config, so nothing has to be
switched off here.

What changes, and why each is forced
-------------------------------------
1. **The robot**, with per-joint torque limits (HAA 24.8, HFE/KFE 48.0) rather
   than one 53.0 across all twelve.
2. **The trunk rename** ``base_link`` -> ``base``, which the mass-randomisation
   event and the contact-sensor path both key on. The sensor path also has to
   gain a level: this asset nests every rigid body at ``Robot/base/<body>``,
   where the stock ``{ENV_REGEX_NS}/Robot/.*`` matches nothing.
3. **Orientation termination instead of contact.** The MJCF gives collision
   geometry to the four feet only, so ``base`` and the hips own no collider and
   ``illegal_contact`` can never fire. Adding bounding-box colliders to them was
   tried and measured worse -- ground support fell from 99.4% to 56% because the
   boxes struck the ground and lifted two feet.
4. **Spawn height** expressed as a clearance above the feet rather than as a
   stance height, expressed as a clearance above the feet, which is what the terrain case needs.

Reward weights, action scale, control rates, network and PPO settings are all
inherited from V8.1 exactly.
"""

from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from guide_robot_v2_cfg import (
    FEET_BELOW_ROOT_FRAME_M,
    GUIDE_ROBOT_V2_CFG,
    GUIDE_V2_BASE_BODY,
)
from v8_1_env_cfg import GuideDutyFloorEnvCfg, V8_1_REWARD_WEIGHTS


# The reward table is untouched; this is a robot change on a flat floor.
V10_FLAT_REWARD_WEIGHTS = dict(V8_1_REWARD_WEIGHTS)

# Tilt past which the trunk counts as fallen. On flat ground nothing legitimate
# comes close, so this is unambiguous.
FALL_TILT_LIMIT_RAD = math.radians(60.0)

# Every rigid body of this asset sits one level deeper than the stock layout.
CONTACT_SENSOR_PATH = "{ENV_REGEX_NS}/Robot/base/.*_foot"

# Drop height for the feet above the floor at reset. Small, because on flat
# ground there is no terrain relief to clear -- only enough to avoid starting
# interpenetrated.
FEET_SPAWN_CLEARANCE_M = 0.05
# Corrected in V10.3. This read ``- FEET_ABOVE_ROOT_FRAME_M`` and produced -0.37,
# which spawned the robot 0.81 m underground; see guide_robot_v2_cfg for the trace.
SPAWN_ROOT_FRAME_Z_M = FEET_SPAWN_CLEARANCE_M + FEET_BELOW_ROOT_FRAME_M


@configclass
class GuideV2FlatEnvCfg(GuideDutyFloorEnvCfg):
    """V8.1's duty-floored Spot recipe, on the 45 kg robot, on flat ground."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = GUIDE_ROBOT_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, SPAWN_ROOT_FRAME_Z_M)

        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=GUIDE_V2_BASE_BODY
        )
        self.scene.contact_forces.prim_path = CONTACT_SENSOR_PATH

        # Contact termination is impossible on this asset; see the docstring.
        self.terminations.body_contact = None
        self.terminations.base_tilt = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": FALL_TILT_LIMIT_RAD},
        )
