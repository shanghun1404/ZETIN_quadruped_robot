"""Stage 1 V9.1 -- stop terminating the episode when a shin brushes the terrain.

The measurement that forced this
--------------------------------
V9 was stopped at iteration 4142 of 8000. It was learning -- mean reward rose
0.40 -> 17.16 and the gait reward 0.035 -> 0.814 -- but the termination
statistics said it could never get anywhere::

                            V8.1 (flat)      V9 (terrain)
    body_contact ends            17.5%            82.5%
    time_out ends (survived)     82.5%            17.5%
    mean episode length        845 / 1000        93 / 1000

The robot died every 1.86 seconds. The consequence is not just a low reward:

    Curriculum/terrain_levels    iter 0: 0.212    iter 4164: 0.180

``terrain_levels_vel`` promotes an environment only once it walks far enough for
its commanded speed, and demotes it when it walks less than half that. Nothing
that lives 1.86 s ever clears the bar, so after 4142 iterations the mean terrain
level had not moved off the easiest row. **The robot would have finished all 8000
iterations without ever seeing a stair**, and the stairs gate would then have
measured a policy trained exclusively on near-flat ground.

The cause
---------
V8's termination set, inherited unchanged through V9::

    body_names="base_link|.*_hip|.*_thigh|.*_calf", threshold=1.0

On flat ground that is exactly right: nothing but a foot has any business
touching the floor, so any other contact is a fall, and terminating protects the
hardware. On rough terrain it is wrong, because a thigh or calf grazing a stair
edge, a box corner or a slope is *ordinary locomotion*, not a crash -- and at a
1.0 N threshold a graze is enough to end the episode.

IsaacLab's own rough-terrain configuration agrees. ``velocity_env_cfg.py``::

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
                "threshold": 1.0},
    )

Body only. The hip, thigh and calf entries were V8's addition, and V8 was a
flat-ground config where that addition cost nothing.

The isolated change
-------------------
The termination body list is narrowed to ``base_link|.*_hip``. Everything else
is inherited from V9 unchanged: the reward table including V8.1's duty floor, the
scanner geometry, the terrain generator and curriculum, the action scale, the
control rates, the network and the PPO settings.

Why the hip stays: ``.*_hip`` is the HAA housing bolted to the body, so it moves
only a few centimetres relative to the trunk. If the hip is loaded, the robot is
genuinely down, and the hardware-protection reason for terminating still holds.
The thigh and calf are the swinging links whose whole job is to move through the
space where terrain is -- those are the two that had to go.

The threshold is deliberately left at 1.0 N. Raising it would have been a second
simultaneous change, and it would blunt the base-contact test as well; if V9.1
still dies too often, the threshold is the next lever, measured on its own.

Falsifiable, and quickly
------------------------
This predicts a specific number moving early: ``Curriculum/terrain_levels`` was
pinned at 0.14-0.21 for 4142 iterations. If it is still under 0.5 by iteration
1500 the diagnosis is wrong and the cause is elsewhere -- most likely the
``base_orientation`` -3.0 and ``base_motion`` -2.0 penalties, which the V9
docstring already flagged as unavoidable to violate on slopes. That check runs
automatically rather than waiting five hours for a final gate.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from v9_env_cfg import GuideTerrainEnvCfg, V9_REWARD_WEIGHTS


# The reward table is untouched; V9.1 is a termination change.
V9_1_REWARD_WEIGHTS = dict(V9_REWARD_WEIGHTS)

# Bodies whose contact with the ground means the robot is down. The thigh and
# calf are excluded because on stairs and boxes they touch during normal
# locomotion; see the module docstring for the measurement.
TERMINATION_BODIES = "base_link|.*_hip"

# Bodies V8 also terminated on, which V9.1 deliberately tolerates.
TOLERATED_CONTACT_BODIES = (".*_thigh", ".*_calf")

# Unchanged from V8/V9 and from stock IsaacLab, so this stays a one-variable edit.
TERMINATION_THRESHOLD_N = 1.0

# If the terrain curriculum has not moved past this by TERRAIN_LEVEL_CHECK_ITER,
# the termination set was not the binding constraint. V9 sat at 0.14-0.21 for
# 4142 iterations, so anything at or below that band is a clear failure.
TERRAIN_LEVEL_CHECK_ITER = 1500
TERRAIN_LEVEL_MIN_EXPECTED = 0.5


@configclass
class GuideTerrainFixedEnvCfg(GuideTerrainEnvCfg):
    """V9's terrain task, with a termination set that terrain can survive."""

    def __post_init__(self):
        super().__post_init__()

        # The single isolated change from V9.
        self.terminations.body_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=TERMINATION_BODIES
                ),
                "threshold": TERMINATION_THRESHOLD_N,
            },
        )
