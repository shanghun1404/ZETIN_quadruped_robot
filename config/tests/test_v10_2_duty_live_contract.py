"""Contract tests for V10.2: the duty floor is switched on, and provably so.

V10.1 ran 3000 iterations with its duty floor masked to exactly zero. The term
was registered at -3.0 and appeared in the weight table the whole time; the
training log shows ``Episode_Reward/duty_floor: 0.0000`` at every logged
iteration, and only two distinct lines in the entire log contain the string.

The resulting contact split reproduced V8's pre-duty-floor failure almost
exactly:

    V8    (no duty floor)   FL 0.161  FR 0.789  RL 0.799  RR 0.162
    V10.1 (floor masked)    FL 0.791  FR 0.248  RL 0.249  RR 0.788

Same signature, opposite diagonal. V10.1 was, in effect, V8's objective again.

These tests pin the three things that let it happen: a mask that read one axis,
a probe that declared a liveness check it no longer performed, and a video tool
whose axis flag reached the camera but not the command.
"""

import ast
from pathlib import Path

import pytest


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _func(module_path: str, name: str) -> str:
    source = _source(module_path)
    tree = ast.parse(source)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(source, node)


# -- the change itself -------------------------------------------------------

def test_v10_2_changes_the_shared_term_and_nothing_in_its_environment():
    """The delta is in v5_12_mdp; the env must be V10.1 verbatim.

    If the config also moved, a change in the duty split could not be attributed
    to the floor switching on.
    """
    source = _source("v10_2_flat_env_cfg.py")
    assert "class GuideV2FlatDutyLiveEnvCfg(GuideV2FlatForwardEnvCfg)" in source
    assert "V10_2_FLAT_REWARD_WEIGHTS = dict(V10_1_FLAT_REWARD_WEIGHTS)" in source
    # No __post_init__ at all: nothing is touched.
    assert "__post_init__" not in source
    for untouched in ("self.rewards", "self.commands", "self.terminations",
                      "self.scene", "self.observations", "self.actions"):
        assert untouched not in source


def test_v10_2_agent_config_is_v10_1_deep_copied_not_restated():
    ppo = _source("v10_2_flat_ppo_cfg.py")
    assert "copy.deepcopy(V10_1_FLAT_PPO_CFG)" in ppo
    assert 'V10_2_FLAT_PPO_CFG.experiment_name = "guide_v10_2_flat_robot_v2"' in ppo
    for forbidden in ("init_noise_std", "actor_hidden_dims", "num_steps_per_env",
                      "learning_rate", "max_iterations"):
        assert forbidden not in ppo


# -- the probe must perform the check its docstring promises -----------------

def test_the_probe_actually_drives_the_duty_estimate_and_requires_a_penalty():
    """probe_v10_1_flat.py looked the term up, then never used it.

    It kept ``duty_col`` and ``term`` as dead assignments while its docstring
    still claimed it "drives the term's internal duty estimate to an
    abandoned-foot value and requires the penalty to actually fire". That check
    exists in probe_v8_1.py and fails on V10.1's configuration, so restoring it
    is what stops this recurring.
    """
    probe = _source("probe_v10_2_flat.py")
    assert "term._duty[:] = 1.0" in probe
    assert "term._duty[:, 0] = 0.0" in probe
    assert 'fired = base.reward_manager._step_reward[:, duty_col]' in probe
    assert "duty floor did not fire on an abandoned foot" in probe
    assert "duty floor produced no penalty on any environment" in probe
    # And the planted case must stay free, or the floor would be a tax on walking.
    assert "duty floor penalizes fully loaded feet" in probe


def test_the_probe_reads_the_moving_mask_directly():
    """The failure was in the mask, so the probe measures the mask, not a proxy."""
    probe = _source("probe_v10_2_flat.py")
    assert "v5_12_mdp._moving_mask(base," in probe
    assert "moving_mask_fraction" in probe
    assert "the duty floor is masked off, which is the V10.1 defect" in probe
    # A command band that never clears the threshold would exempt the term just
    # as completely as the wrong axis did.
    assert "the duty floor would be exempt at the low end" in probe


def test_the_dead_liveness_assignments_are_not_carried_forward():
    """A lookup with no assertion after it is worse than no lookup at all."""
    probe = _source("probe_v10_2_flat.py")
    assert probe.count("duty_col") >= 2, "duty_col must be used, not just assigned"
    body = probe.split("duty_col = ")[1]
    assert "_step_reward[:, duty_col]" in body


# -- the video tool's axis flag must reach the command -----------------------

def test_video_command_honours_the_forward_axis_not_just_the_camera():
    """--forward-axis +y placed the camera but left the command on base -x.

    The V10.1 videos were therefore filmed from the correct side while the robot
    was asked to travel along an axis its policy had only ever seen pinned to
    zero. The gate numbers were unaffected -- evaluate_v8.py branches correctly
    -- but the footage did not show what it claimed to.
    """
    video = _source("play_v8_video.py")
    assert 'parser.add_argument("--forward-axis", choices=("-x", "+y")' in video
    block = video.split("--speed is a guide-frame forward speed")[1]
    assert 'if args_cli.forward_axis == "+y":' in block
    assert "cmd.ranges.lin_vel_x = (0.0, 0.0)" in block
    assert "cmd.ranges.lin_vel_y = (args_cli.speed, args_cli.speed)" in block
    # The -x path must survive unchanged so older checkpoints still film.
    assert "guide_forward_speed_range_to_base(" in block


def test_the_video_tool_and_the_evaluator_agree_on_what_plus_y_means():
    """Two tools disagreeing about the axis is how the footage and the gate diverged."""
    video = _source("play_v8_video.py")
    evaluator = _source("evaluate_v8.py")
    for source in (video, evaluator):
        assert "cmd.ranges.lin_vel_x = (0.0, 0.0)" in source
        assert "cmd.ranges.lin_vel_y = (args_cli.speed, args_cli.speed)" in source
    # And the evaluator must read speed off the same axis it commands.
    assert "robot.data.root_lin_vel_b[:, 1]" in evaluator


# -- tooling targets V10.2 without stranding older checkpoints ---------------

def test_v10_2_tooling_targets_v10_2(tmp_path=None):
    queue = _source("queue_v10_2_flat_pilot.sh")
    assert "probe_v10_2_flat.py" in queue and "train_v10_2_flat.py" in queue
    assert "logs/rsl_rl/guide_v10_2_flat_robot_v2" in queue
    assert queue.count("--env v10_2_flat") >= 4
    # The axis flag has to reach every rollout, or the run is scored on x again.
    assert queue.count("--forward-axis +y") >= 4


def test_older_checkpoints_stay_evaluable():
    for name in ("evaluate_v8.py", "play_v8_video.py"):
        source = _source(name)
        for older in ("v10_flat", "v10_1_flat"):
            assert older in source, f"{older} checkpoints must stay evaluable"


@pytest.mark.parametrize(
    "path",
    ["v10_2_flat_env_cfg.py", "v10_2_flat_ppo_cfg.py", "train_v10_2_flat.py",
     "probe_v10_2_flat.py"],
)
def test_v10_2_sources_parse(path):
    ast.parse(_source(path))
