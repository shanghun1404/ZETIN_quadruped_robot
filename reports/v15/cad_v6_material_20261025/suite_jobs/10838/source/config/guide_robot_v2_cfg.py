"""The new 45 kg quadruped, converted from MuJoCo MJCF and verified in PhysX.

How this asset differs from ``guide_robot_cfg``
-----------------------------------------------
======================  =====================  =====================
                        v1 (guide_quadruped)   v2 (quadruped.xml)
======================  =====================  =====================
source format           URDF                   **MJCF**
links / bodies          18                     **54**
total mass              25.34 kg               **45.451 kg**
trunk link name         ``base_link``          **``base``**
joint names             FL/FR/RL/RR HAA/HFE/KFE  same
HAA torque limit        53.0 N m               **24.8 N m**
HFE / KFE torque limit  53.0 N m               **48.0 N m**
======================  =====================  =====================

The torque-to-weight ratio is roughly halved, so nothing measured on v1 -- the
V8.1 duty floor, the V9.2 yaw geometry, the V9.3 kernel width, the V9.4 speed
ramp -- can be assumed to transfer. Those are hypotheses again on this robot.

Conversion and repair, all measured
-----------------------------------
``MjcfConverter`` needs ``isaacsim.asset.importer.mjcf`` enabled explicitly on
Isaac Sim 5.1, or ``MJCFCreateImportConfig`` is unregistered and the import fails
with a NoneType import config.

The converted stage then needed two repairs, and one apparent defect turned out
not to be one:

* **Two articulation roots** (``/quadruped/base/base`` and
  ``/quadruped/worldBody``). IsaacLab requires one. The wrapper's root API was
  removed.
* **The MJCF world plane was baked into the robot asset** at
  ``/quadruped/worldBody/_/collisions/CollisionPlane``. Being a sibling of
  ``/quadruped/base``, the spawn transform carried it with the robot -- the robot
  stood on a floor it brought along, and never fell. ``worldBody`` is
  deactivated. Per environment this would also have been 4096 infinite planes.
* **"No collision geometry" was a false alarm.** Every body's ``visuals`` and
  ``collisions`` scope is *instanced*, and a plain ``Stage.Traverse()`` does not
  descend into instances -- it reports zero meshes and one collider. With
  ``Usd.TraverseInstanceProxies()`` the feet have their colliders exactly as the
  MJCF asks (``<default><geom contype="0" conaffinity="0"/></default>`` plus
  ``contype=1`` on the four ``*_foot_geom``).

Verified standing, not assumed
------------------------------
Per-body frame positions cannot verify this asset, because the MJCF conversion
leaves them offset from their geometry -- a foot frame can read above the trunk.
Contact force does not care about frame bookkeeping, so that is what was
measured::

**Corrected in V10.3:** this paragraph previously read "at rest the root frame
reads z = -0.419". That was wrong by sign and, worse, it applied the per-body
frame argument to the articulation *root*, which the spawn trace disproves: the
root reads exactly ``init_state.pos.z`` before the first step and settles at
+0.445. The distinction cost four versions of underground spawns.

    per-foot vertical force   108.4 / 110.1 / 116.3 / 108.4 N
    total                     443.2 N
    own weight                445.9 N   ->  99.4% supported, all four feet

An earlier repair attempt added bounding-box colliders to ``base`` and the hips
for contact-based termination. Those boxes were placed from bounds that include
the baked offsets, so they struck the ground, tilted the robot and lifted two
feet -- support fell to 56% with FL and RL at 0 N. They are not used. Termination
is handled by body orientation instead, which is immune to the frame offset; see
``v10_env_cfg``.

Consequences for anything that reads absolute height
-----------------------------------------------------
``root_pos_w[:, 2]`` is offset by a constant and is **not** a stance height on
this robot. ``evaluate_v9`` already reports base height relative to the nearest
height-scanner ground hit, which is a difference and therefore immune. No reward
term in the V8.1-V9.4 table uses absolute height.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg


# Repaired asset: worldBody deactivated, single articulation root, foot colliders
# intact. Produced by repair_v2_usd.py and verified by the contact test above.
GUIDE_V2_USD_PATH = "/home/kang/robot_v2/usd/v2_plain.usd"

# The trunk link. v1 called it base_link; every body-name pattern in the config
# chain has to follow this rename.
GUIDE_V2_BASE_BODY = "base"

# The MJCF's own ``stand`` keyframe: qpos base z 0.4640, per leg (0, -0.55, 1.10).
STAND_HFE_RAD = -0.55
STAND_KFE_RAD = 1.10

# Root frame z at which the robot rests holding the stand pose, with the ground
# carrying its weight. Measured by dropping it and reading root_pos_w[:, 2].
#
#   probe_v10_1_flat.py   +0.4357   64 envs, 400 steps, support 0.936
#   spawn_trace_probe.py  +0.4451   64 envs, 400 steps, support 0.949
#   probe_v10_2_flat.py   +0.4455   64 envs, 400 steps, support 0.952
#
# The ~1 cm spread is the base-mass randomisation changing how far the robot
# settles into the contact; 0.445 is taken from the two most controlled runs.
SETTLED_ROOT_FRAME_Z_M = 0.445

# Distance from the root frame DOWN to the feet. Same measurement: the feet rest
# on the ground (z = 0) while the frame reads +0.445.
FEET_BELOW_ROOT_FRAME_M = 0.445

# WRONG UNTIL V10.3, AND IT COST TWO RUNS.
#
# These previously read ``SETTLED_ROOT_FRAME_Z_M = -0.4192`` and
# ``FEET_ABOVE_ROOT_FRAME_M = 0.42`` -- the right magnitude with the sign
# inverted, and both annotated as measured. Every spawn height derived from them
# as ``clearance - 0.42`` was therefore low by about ``2 x 0.44``.
#
# ``spawn_trace_probe.py`` settled it. Spawning at the configured flat value of
# -0.37 and stepping with zero action:
#
#   before any step  root z -0.3700   (exactly init_state.pos.z -- no hidden offset)
#   step 1                  -0.3651   ground support 2.432 x body weight
#   step 20                 -0.0018   0.979
#   step 200                +0.4328   0.949
#   step 400                +0.4451   0.949
#
# The robot spawned 0.81 m underground, took a 2.43x-body-weight depenetration
# impulse and rose 0.815 m. On a flat GroundPlane the solver ejects it and it
# settles correctly, which is why every flat gate passed with this defect live.
# It is visible in the V10.1 and V10.2 videos: the first ~14 frames show the
# robot rising out of the floor.
#
# The trace also disproves the reasoning that produced -0.4192. ``init_state.pos.z``
# IS what ``root_pos_w[:, 2]`` reports for the root body; the frame displacement
# this asset carries affects per-body positions, not the articulation root.

# Spawn clearance for generated terrain.
#
# On terrain a robot spawns at ``env_origin + init_state.pos``, and each tile's
# origin sits at its own local surface -- a box tile is raised, a stairs tile is
# a pit. The intent behind 0.32 m survives the sign fix: it is enough clearance
# for low-difficulty relief and a short enough drop to land upright. What was
# wrong was the arithmetic under it.
#
# Under the old sign this value produced a frame spawn of -0.10, which was
# believed to put the feet 0.32 m above the origin and in fact put them 0.545 m
# below it. The two recorded terrain failures follow from that: spawning the
# frame at -0.389 made the robot "fall through the mesh from underneath" (feet
# 0.83 m below the origin, root z running -0.324 -> -5.4 -> -45 -> -314 with zero
# contact force), and every spawn high enough to make contact landed hard.
#
# UNVALIDATED FOR TERRAIN. The predecessor report records a +0.6 m frame spawn
# that made contact and still flipped to 165.6 degrees, and the corrected value
# below is +0.765 -- close enough that the sign fix alone may not resolve
# terrain. A raised tile can still put its surface above the feet at this
# height. V10.3 is flat-only; re-validate this constant with a terrain-side
# trace before trusting it.
FEET_SPAWN_CLEARANCE_M = 0.32
SPAWN_ROOT_FRAME_Z_M = FEET_SPAWN_CLEARANCE_M + FEET_BELOW_ROOT_FRAME_M

# Per-joint limits from the MJCF actuator block. v1 applied a single 53.0 N m to
# every joint; this robot's hip abduction is rated well below its other two.
HAA_EFFORT_LIMIT_NM = 24.8
HFE_KFE_EFFORT_LIMIT_NM = 48.0

GUIDE_ROBOT_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GUIDE_V2_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 54 bodies with nested cosmetic panels overlap heavily by design.
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SPAWN_ROOT_FRAME_Z_M),
        joint_pos={
            ".*_HAA": 0.0,
            ".*_HFE": STAND_HFE_RAD,
            ".*_KFE": STAND_KFE_RAD,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_abduction": DCMotorCfg(
            joint_names_expr=[".*_HAA"],
            effort_limit=HAA_EFFORT_LIMIT_NM,
            saturation_effort=HAA_EFFORT_LIMIT_NM,
            velocity_limit=50.0,
            stiffness=300.0,
            damping=8.0,
            friction=0.3,
        ),
        "hip_knee": DCMotorCfg(
            joint_names_expr=[".*_HFE", ".*_KFE"],
            effort_limit=HFE_KFE_EFFORT_LIMIT_NM,
            saturation_effort=HFE_KFE_EFFORT_LIMIT_NM,
            velocity_limit=50.0,
            stiffness=300.0,
            damping=8.0,
            friction=0.3,
        ),
    },
)
