"""Stage 1 V7: V6 rewards with the action scale the working reference used.

Why this variable, and why now
------------------------------
V5.11 through V6 changed the reward four times. Across all of them the front
pair's HFE correlation moved a total of 0.02 (+0.698 -> +0.676) while the rear
pair went all the way to -0.404 under the *same* penalty. Four reward edits on a
stuck axis is enough evidence that the reward is not the binding constraint.

One variable has never been touched: **action scale**. The `figure8_v6`
reference model -- the one that actually produced smooth, natural, commanded
locomotion on this robot -- used `scale = 0.05`, five times smaller than the
0.25 the whole V5 line inherited. Its own 2026-07-13 analysis already
recommended this:

    "기준 모델처럼 gait를 덜 규정하고 action 범위를 줄이는 방향을
     먼저 계측할 가치가 있다."

That recommendation was never executed; V5.6 through V6 kept editing rewards
instead. V7 executes it.

0.10 rather than 0.05: the reference also ran a different network, stance and
motor model, so jumping straight to its value would confound the comparison with
V6. 0.10 is a 2.5x reduction -- large enough to move the outcome if action
authority is the constraint, small enough that a failure still says something.

Everything else is inherited from V6 unchanged, so any change is attributable to
the action scale alone.

Downstream note
---------------
Action scale is part of the deployment contract: the observation feeds back
`(target - default) / scale` at `proprio[30:42]`, and `guide_locomotion/runtime.py`
hard-codes `ACTION_SCALE = 0.25`. If V7 wins, that constant and its parity test
must move to 0.10 before anything is deployed.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from v6_env_cfg import UprightAntiCheatGuideEnvCfg, V6_REWARD_WEIGHTS


ACTION_SCALE = 0.10

# The reward table is untouched; V7 is an action-authority experiment.
V7_REWARD_WEIGHTS = dict(V6_REWARD_WEIGHTS)


@configclass
class SmallActionGuideEnvCfg(UprightAntiCheatGuideEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # The single isolated change from V6.
        self.actions.joint_pos.scale = 0.10
