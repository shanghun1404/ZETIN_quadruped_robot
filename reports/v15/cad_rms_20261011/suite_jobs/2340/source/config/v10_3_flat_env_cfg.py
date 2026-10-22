"""Stage 1 V10.3-flat -- V10.2 with the robot spawned above the floor instead of inside it.

The measured defect
-------------------
Every run on this robot -- V10, V10-flat, V10.1, V10.2 -- spawned it 0.81 m
underground. ``spawn_trace_probe.py``, 64 environments, zero action, flat
GroundPlane::

    before any step   root z -0.3700   exactly init_state.pos.z
    step 1                   -0.3651   ground support 2.432 x body weight
    step 2                   -0.3534   2.093
    step 5                   -0.2958   1.060
    step 20                  -0.0018   0.979
    step 50                  +0.3848   0.771
    step 200                 +0.4328   0.949
    step 400                 +0.4451   0.949

The robot appears where the config puts it, absorbs a depenetration impulse of
2.43 times its own weight, and rises 0.815 m before settling. It is visible in
the V10.1 and V10.2 recordings: the first frames show empty floor, then the
trunk emerging through it.

Root cause
----------
``guide_robot_v2_cfg.py`` declared ``FEET_ABOVE_ROOT_FRAME_M = 0.42`` and
``SETTLED_ROOT_FRAME_Z_M = -0.4192`` -- the right magnitudes with the signs
inverted, both annotated as measured. The feet are 0.445 m *below* the root
frame. Every spawn derived as ``clearance - 0.42`` was low by about ``2 x 0.44``.

The trace also disproves the argument that produced the negative value. The
robot reads exactly -0.3700 before the first step, so ``init_state.pos.z`` is
precisely what ``root_pos_w[:, 2]`` reports: the frame displacement this asset
carries affects per-body positions, not the articulation root. Reasoning that
was correct about ``body_pos_w`` was applied to a quantity it does not govern.

The isolated change
-------------------
``SPAWN_ROOT_FRAME_Z_M = clearance + FEET_BELOW_ROOT_FRAME_M``, which for flat
ground's 0.05 m clearance gives **+0.495** instead of -0.37. Nothing else moves:
same robot, same fifteen reward weights, same command band on +y, same
termination set, same action scale, same control rates, same PPO settings, same
seed. As with V10.2 the delta lives outside this file, so the class below is
deliberately empty.

Why this is a separate version from V10.2
-----------------------------------------
V10.2 answered whether a live duty floor produces an even duty split, and it did
-- 8/8 gates at iterations 1000, 2000 and 2999, contact fractions within two
points of even on all four feet. Folding the spawn fix into that run would have
made the duty result unattributable. It is also a genuinely separate question:
V10.2's gait was learned *while* starting each episode underground, so what this
run measures is what the same objective produces without that.

What to watch
-------------
1. **Whether the gait changes at all.** V10.2 already passed every gate. If the
   numbers barely move, the depenetration transient was cosmetic; if survival,
   duty or phase shift, it was shaping the policy.
2. **The phase drift V10.2 showed.** Its rear-pair correlation went -0.1906 ->
   +0.1229 -> +0.1786 across iterations 1000/2000/2999 while duty stayed flat,
   so ``model_1000`` was the better trot. Whether that drift persists here is the
   more interesting result than the gate verdict.
3. **Gate contamination.** ``evaluate_v8.py`` samples from step 0 with no warmup
   discard. Under zero action depenetration takes ~200 steps; in the V10.2 video
   the trained policy is up within roughly 14 frames, so the contamination was
   probably small -- but it was never measured. ``--warmup-seconds`` now exists
   for exactly that, defaulting to 0.0 so no archived number changes.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from guide_robot_v2_cfg import FEET_BELOW_ROOT_FRAME_M
from v10_flat_env_cfg import FEET_SPAWN_CLEARANCE_M, SPAWN_ROOT_FRAME_Z_M
from v10_2_flat_env_cfg import (
    FORWARD_AXIS,
    FORWARD_SIGN,
    GuideV2FlatDutyLiveEnvCfg,
    V10_2_FLAT_REWARD_WEIGHTS,
)


# Unchanged from V10.2 -- the fix is in the spawn arithmetic, not the objective.
V10_3_FLAT_REWARD_WEIGHTS = dict(V10_2_FLAT_REWARD_WEIGHTS)

# Re-exported so the probe can assert against them without importing three modules.
V10_3_SPAWN_ROOT_FRAME_Z_M = SPAWN_ROOT_FRAME_Z_M
V10_3_FEET_SPAWN_CLEARANCE_M = FEET_SPAWN_CLEARANCE_M
V10_3_FEET_BELOW_ROOT_FRAME_M = FEET_BELOW_ROOT_FRAME_M


@configclass
class GuideV2FlatSpawnFixedEnvCfg(GuideV2FlatDutyLiveEnvCfg):
    """V10.2's objective, with the robot starting above the floor rather than in it.

    Deliberately empty, exactly as V10.2's class is. The spawn height is derived
    in ``v10_flat_env_cfg`` from a constant in ``guide_robot_v2_cfg``, and both
    were corrected in place because they were factually wrong -- re-running an
    older config now spawns correctly and will therefore not reproduce the
    archived V10.1 / V10.2 numbers. That is recorded in the report rather than
    worked around, because leaving a known-wrong constant in place to preserve
    reproducibility would guarantee the next version inherits the bug.
    """
