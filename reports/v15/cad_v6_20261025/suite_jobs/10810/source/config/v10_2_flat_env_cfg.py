"""Stage 1 V10.2-flat -- V10.1 with the duty floor actually switched on.

The measured defect
-------------------
V10.1 pointed the command along the robot's real fore-aft axis and the gait
became a genuine diagonal trot: the FL-RR correlation went from -0.020 under the
crab walk to +0.725, FR-RL to +0.454, and both same-end pairs stayed out of
phase. Seven of the eight gait gates passed.

The eighth did not, and its numbers are lopsided::

    foot contact fraction   FL 0.791   FR 0.248   RL 0.249   RR 0.788
    min_foot_contact        0.248 against a >= 0.30 gate

One diagonal carries the robot for four fifths of the cycle while the other
spends three quarters of it in the air. That is the exact failure the duty floor
was added in V8.1 to prevent, and the training log says the duty floor never
once fired::

    Episode_Reward/duty_floor: 0.0000     at every logged iteration

Across the whole V10.1 log only two distinct lines contain "duty_floor": the
weight-table row (-3.0) and that zero. The same grep over V10-flat's log -- same
robot, same objective, command on x -- returns 565 distinct values.

Root cause
----------
``v5_12_mdp._moving_mask`` decided "is the robot being asked to move?" by reading
a single column::

    command = smooth_mdp.guide_velocity_command(env, command_name)[:, 0]
    return torch.abs(command) >= moving_threshold

``guide_velocity_command`` is the base command passed through
``guide_frame.base_to_guide_observation``, which is ``_rotate_planar_180`` --
it negates x and y and never swaps them, so column 0 is still the x channel.
V10.1 commands ``lin_vel_x = (0.0, 0.0)``, so that column is identically zero,
the mask is False for every environment at every step, and
``worst_foot_contact_deficit`` multiplies its deficit by that mask. The term was
registered at -3.0 and returned zero for three thousand iterations.

This is the second hidden "forward is base x" assumption to cost a run. The
first put the command on the wrong axis and produced the crab walk; this one
silently disabled the term that would have shaped the gait once the axis was
fixed. Both were invisible to every gate, because a masked-off reward term does
not report itself as absent -- it reports as satisfied.

The isolated change
-------------------
``_moving_mask`` now tests the planar command magnitude instead of one column,
which is axis-independent. Every configuration in the repository older than
V10.1 pins the lateral band to exactly zero -- ``smooth_env_cfg.py`` line 192 for
the V5 line, ``v8_env_cfg.py`` line 151 for the V8 line -- so on those the norm
of ``(x, 0)`` equals ``|x|`` and the fix is numerically a no-op. It changes
behaviour on V10.1's command and nowhere else.

Nothing in this file's environment differs from V10.1: same robot, same fifteen
reward weights, same command band on +y, same termination set, same action scale
and control rates, same PPO settings. The delta lives in the shared term, which
is why this is a new version rather than a re-run -- the code that produced
V10.1's numbers no longer exists.

What the probe now proves
-------------------------
``probe_v8_1.py`` drove the term's internal duty estimate to an abandoned-foot
value and required the penalty to fire. That check was dropped when the probe
was ported to the new robot: ``probe_v10_flat.py`` and ``probe_v10_1_flat.py``
keep the line that looks the term up and then never use it, while their
docstrings still claim the check is there. Restored here, it fails on V10.1's
configuration -- so the run would not have launched.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from v10_1_flat_env_cfg import (
    FORWARD_AXIS,
    FORWARD_SIGN,
    GuideV2FlatForwardEnvCfg,
    V10_1_FLAT_REWARD_WEIGHTS,
    V10_1_FORWARD_SPEED_RANGE,
    V10_1_LATERAL_SPEED_RANGE,
)


# Unchanged from V10.1 -- the fix is in ``v5_12_mdp._moving_mask``, not here.
V10_2_FLAT_REWARD_WEIGHTS = dict(V10_1_FLAT_REWARD_WEIGHTS)

V10_2_FORWARD_SPEED_RANGE = V10_1_FORWARD_SPEED_RANGE
V10_2_LATERAL_SPEED_RANGE = V10_1_LATERAL_SPEED_RANGE


@configclass
class GuideV2FlatDutyLiveEnvCfg(GuideV2FlatForwardEnvCfg):
    """V10.1's environment, run against a duty floor that is no longer masked off.

    Deliberately empty. Adding a second change here would make it impossible to
    say whether a change in the duty split came from the floor switching on or
    from whatever else was adjusted alongside it.
    """
