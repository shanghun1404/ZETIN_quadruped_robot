"""Contract tests for V9.3: a velocity kernel that discriminates.

V9.2's yaw band worked -- the curriculum climbed to 2.111 by iteration 1479 --
and then collapsed to 0.265 because the policy converged on standing still.
Across every sampled checkpoint it travelled 0.085-0.156 m/s against a commanded
0.500, never climbed a stair, and survived 100% of episodes by not trying.

The cause is arithmetic on an exponential kernel, so most of these tests are
arithmetic on that kernel.
"""

import ast
import importlib.util
import math
from pathlib import Path

import pytest


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _const(name, module="v9_3_env_cfg.py"):
    tree = ast.parse(_source(module))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {module}")


def _standing_pays(command, std, weight=5.0):
    """exp(-error/std) with error == command, which is what standing still gives."""
    return weight * math.exp(-abs(command) / std)


def _gradient(command, std, weight=5.0):
    """Magnitude of d(reward)/d(error) at the operating error."""
    return (weight / std) * math.exp(-abs(command) / std)


def test_v9_3_narrows_one_kernel_and_changes_nothing_else():
    source = _source("v9_3_env_cfg.py")
    assert "class GuideTerrainSharpVelEnvCfg(GuideTerrainTurnBandEnvCfg)" in source
    assert source.count("self.rewards.") == 1
    assert 'self.rewards.base_linear_velocity.params["std"] = V9_3_LIN_VEL_STD' in source
    for untouched in ("self.commands", "self.terminations", "self.scene",
                      "self.observations", "self.curriculum", "self.actions"):
        assert untouched not in source
    # The weight table is inherited verbatim -- only the kernel width moves.
    assert "V9_3_REWARD_WEIGHTS = dict(V9_2_REWARD_WEIGHTS)" in source
    assert "weight" not in source.split("def __post_init__")[1]


def test_the_kernel_narrows_rather_than_widens():
    assert _const("V9_2_LIN_VEL_STD") == 1.0
    assert _const("V9_3_LIN_VEL_STD") == 0.5
    assert _const("V9_3_LIN_VEL_STD") < _const("V9_2_LIN_VEL_STD")


def test_standing_still_pays_substantially_less_than_it_did():
    """The measured defect: V9.2 collected 2.99/s from the velocity term at rest."""
    c = _const("OPERATING_SPEED_M_S")
    before = _standing_pays(c, _const("V9_2_LIN_VEL_STD"))
    after = _standing_pays(c, _const("V9_3_LIN_VEL_STD"))

    # V9.2's measured 2.99/s is what this formula predicts for a stationary robot.
    assert before == pytest.approx(2.74, abs=0.05)
    # The new kernel must cut that meaningfully, not shave it.
    assert after < 0.65 * before
    assert after == pytest.approx(1.50, abs=0.05)


def test_the_chosen_width_also_raises_the_gradient_the_policy_feels():
    """A sharper kernel usually flattens far from target; 0.5 does not, yet.

    This is the reason for 0.5 rather than something sharper: it is the widest
    value that both cuts the standing payoff and increases the local gradient at
    the operating error, so it cannot stop teaching a policy that starts at rest.
    """
    c = _const("OPERATING_SPEED_M_S")
    old_std = _const("V9_2_LIN_VEL_STD")
    new_std = _const("V9_3_LIN_VEL_STD")

    assert _gradient(c, new_std) > _gradient(c, old_std)
    # And sharper choices would lose that property, which is why they were not taken.
    assert _gradient(c, 0.25) < _gradient(c, old_std)


def test_moving_out_earns_standing_under_the_new_kernel():
    c = _const("OPERATING_SPEED_M_S")
    weight = 5.0
    standing = _standing_pays(c, _const("V9_3_LIN_VEL_STD"), weight)
    moving = weight - standing
    assert moving > standing
    # Under V9.2's kernel it was the other way round, which is the whole problem.
    old_standing = _standing_pays(c, _const("V9_2_LIN_VEL_STD"), weight)
    assert (weight - old_standing) < old_standing


def test_helper_matches_the_spot_kernel_it_models():
    source = _source("v9_3_env_cfg.py")
    assert "def standing_still_reward_per_s" in source
    assert "math.exp(-abs(command_m_s) / std)" in source
    # And it must refuse a nonsensical width rather than dividing by zero.
    assert "if std <= 0.0:" in source


def test_probe_reads_the_std_back_out_of_the_reward_manager():
    probe = _source("probe_v9_3.py")
    assert 'base.reward_manager.get_term_cfg("base_linear_velocity")' in probe
    assert 'vel_cfg.params["std"]' in probe
    assert "standing_pays_v9_2" in probe and "standing_pays_v9_3" in probe
    assert "moving must out-earn standing" in probe
    # The published weight must not have moved as a side effect.
    assert "velocity weight must stay at the published 5.0" in probe
    # Everything inherited through four levels still has to be re-checked.
    assert "V8.1 duty floor lost" in probe
    assert "V9.1 termination fix lost" in probe
    assert "terrain_levels curriculum missing" in probe


def test_the_curriculum_check_runs_for_the_whole_job_not_just_once():
    """V9.2's check passed at 1501 and the collapse happened afterwards, unseen."""
    queue = _source("queue_v9_3_pilot.sh")
    assert "check_terrain_progress.py" in queue
    assert "v9_3_terrain_ALERT.json" in queue
    assert '"passed": true' in queue
    # The `&& break` that stopped watching after the first verdict is gone.
    assert "--quiet && break" not in queue
    assert 'kill -0 "$TRAIN_PID"' in queue


def test_alert_would_have_fired_on_v9_2s_actual_series():
    """The regression test: run V9.2's measured curve through the live checker."""
    spec = importlib.util.spec_from_file_location("ctp", "check_terrain_progress.py")
    ctp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctp)

    v9_2 = [(0, 0.987), (500, 1.583), (1000, 1.500), (1479, 2.111), (1501, 2.103),
            (2000, 1.417), (2500, 0.797), (3000, 0.401), (4999, 0.265)]

    # At the original check point it passed, and correctly so.
    early = ctp.verdict(v9_2[:5], 1500, ctp.TERRAIN_LEVEL_MIN_EXPECTED,
                        ctp.TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK)
    assert early["passed"] is True

    # By 2500 the decline is unmistakable, and that is when the alert must fire.
    at_2500 = ctp.verdict(v9_2[:7], 2500, ctp.TERRAIN_LEVEL_MIN_EXPECTED,
                          ctp.TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK)
    assert at_2500["passed"] is False
    assert at_2500["declining"] is True
    assert at_2500["peak_level"] == pytest.approx(2.111)
    assert at_2500["decline_from_peak"] > 0.5

    # And at the end it is still failing, not silently recovered.
    final = ctp.verdict(v9_2, 4999, ctp.TERRAIN_LEVEL_MIN_EXPECTED,
                        ctp.TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK)
    assert final["passed"] is False


def test_error_vel_xy_is_not_used_as_a_speed_proxy():
    """It cannot tell walking from standing on terrain; the numbers are in the docstring.

    V8.1 flat walking 0.394, V9.1 terrain walking 1.073, V9.2 terrain stationary
    1.151 -- the two terrain runs are 0.08 apart with opposite behaviour.
    """
    checker = _source("check_terrain_progress.py")
    assert "error_vel_xy" not in checker
    # And the rejection must be recorded where the next person will look.
    assert "error_vel_xy" in _source("v9_3_env_cfg.py")


def test_v9_3_tooling_targets_v9_3_but_keeps_older_versions_evaluable():
    queue = _source("queue_v9_3_pilot.sh")
    assert "probe_v9_3.py" in queue and "train_v9_3.py" in queue
    assert "logs/rsl_rl/guide_v9_3_terrain" in queue
    assert queue.count("--env v9_3") >= 6
    for name in ("evaluate_v9.py", "play_v9_video.py"):
        source = _source(name)
        # Later versions extend the choice list and move the default forward;
        # what V9.3 owns is that its own checkpoints stay evaluable under the
        # config they trained with. Survival is a gate and the termination set
        # decides it, so scoring a checkpoint under a newer config misreports it.
        assert '"v9_3"' in source
        assert "from v9_3_env_cfg import GuideTerrainSharpVelEnvCfg" in source
        for older in ("v9_1_env_cfg", "v9_2_env_cfg"):
            assert older in source, "older checkpoints must stay evaluable"


def test_v9_3_agent_config_is_v9_2_deep_copied_not_restated():
    ppo = _source("v9_3_ppo_cfg.py")
    assert "copy.deepcopy(V9_2_PPO_CFG)" in ppo
    assert 'V9_3_PPO_CFG.experiment_name = "guide_v9_3_terrain"' in ppo
    for forbidden in ("init_noise_std", "actor_hidden_dims", "num_steps_per_env", "learning_rate"):
        assert forbidden not in ppo


@pytest.mark.parametrize(
    "path", ["v9_3_env_cfg.py", "v9_3_ppo_cfg.py", "train_v9_3.py", "probe_v9_3.py"]
)
def test_v9_3_sources_parse(path):
    ast.parse(_source(path))
