"""Contract tests for V9.2: a yaw band the terrain curriculum can reward.

V9.1 fixed survival completely (episode length 93 -> 1000 of 1000, body contact
82.5% -> 0.05%) and its terrain curriculum still went backwards: it peaked at
1.584 around iteration 500 and fell to 0.716 by 2000, out of 10 levels. The
cause is arithmetic, so most of these tests are arithmetic.
"""

import ast
import importlib.util
import math
from pathlib import Path

import pytest


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _const(name, module="v9_2_env_cfg.py"):
    tree = ast.parse(_source(module))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {module}")


def _chord(v, w, t):
    """Local copy of the geometry, so the test does not import the module it checks."""
    if w == 0.0:
        return v * t
    return 2.0 * (v / abs(w)) * abs(math.sin(abs(w) * t / 2.0))


def test_v9_2_narrows_the_yaw_band_and_changes_nothing_else():
    source = _source("v9_2_env_cfg.py")
    assert "class GuideTerrainTurnBandEnvCfg(GuideTerrainFixedEnvCfg)" in source
    assert source.count("self.commands.") == 1
    assert "self.commands.base_velocity.ranges.ang_vel_z = V9_2_YAW_RATE_RANGE" in source
    for untouched in ("self.rewards", "self.terminations", "self.scene", "self.observations", "self.curriculum"):
        assert untouched not in source
    assert "V9_2_REWARD_WEIGHTS = dict(V9_1_REWARD_WEIGHTS)" in source


def test_yaw_band_is_narrower_than_v9_1_but_still_allows_turning():
    low, high = _const("V9_2_YAW_RATE_RANGE")
    assert (low, high) == (-0.2, 0.2)
    # V8/V9/V9.1 used +/-0.6.
    assert high < 0.6
    # A guide dog has to turn on command; zero would be a different task.
    assert high > 0.0


def test_heading_command_stays_off_so_the_policy_still_takes_yaw_commands():
    """The stock fix is heading_command=True; it would remove commanded yaw."""
    assert "heading_command=False" in _source("v8_env_cfg.py")
    assert "heading_command" not in _source("v9_2_env_cfg.py").split('"""')[2]
    # The probe must enforce it at runtime, since it is inherited not restated.
    assert "heading_command must stay off" in _source("probe_v9_2.py")


def test_promotion_arithmetic_matches_the_curriculum_source():
    """terrain_levels_vel promotes on size[0]/2 and ROUGH_TERRAINS_CFG tiles are 8 m."""
    assert _const("TERRAIN_TILE_SIZE_M") == 8.0
    assert _const("EPISODE_LENGTH_S") == 20.0
    # PROMOTION_DISTANCE_M is TERRAIN_TILE_SIZE_M / 2.0, an expression rather
    # than a literal, so check the arithmetic and the source separately.
    assert "PROMOTION_DISTANCE_M = TERRAIN_TILE_SIZE_M / 2.0" in _source("v9_2_env_cfg.py")
    assert _const("TERRAIN_TILE_SIZE_M") / 2.0 == 4.0
    # And the config must actually run 20 s episodes, or the arithmetic is wrong.
    assert "self.episode_length_s = 20.0" in _source("v8_env_cfg.py")


def test_the_chosen_band_actually_clears_the_promotion_bar():
    """This is the whole justification for 0.2 rather than any other number."""
    low, high = _const("V9_2_YAW_RATE_RANGE")
    promote = _const("TERRAIN_TILE_SIZE_M") / 2.0
    episode = _const("EPISODE_LENGTH_S")
    worst_yaw = max(abs(low), abs(high))

    # V8's forward band, in guide-frame speeds.
    slow, fast = 0.3, 0.9
    assert "V8_GUIDE_FORWARD_SPEED_RANGE = (0.3, 0.9)" in _source("v8_env_cfg.py")

    # At full speed, even a worst-case sustained turn must promote.
    assert _chord(fast, worst_yaw, episode) > promote
    # Walking straight must promote at any commanded speed.
    assert _chord(slow, 0.0, episode) > promote
    # And the band V9.1 used must fail this same test -- otherwise the change
    # is not addressing the measured problem.
    assert _chord(fast, 0.6, episode) < promote


def test_chord_helper_agrees_with_the_closed_form():
    spec = importlib.util.spec_from_file_location("v92", "v9_2_env_cfg.py")
    # The module imports isaaclab; only the pure helper is under test, so compare
    # the source's formula against the local one rather than importing.
    source = _source("v9_2_env_cfg.py")
    assert "def displacement_chord_m" in source
    assert "2.0 * radius * abs(math.sin(" in source
    assert "radius = speed_m_s / abs(yaw_rate_rad_s)" in source
    # Straight-line case must be handled separately, not by dividing by zero.
    assert "if yaw_rate_rad_s == 0.0:" in source
    assert "return speed_m_s * seconds" in source
    assert spec is not None


def test_probe_runs_the_arithmetic_on_resolved_values_not_the_config_literal():
    probe = _source("probe_v9_2.py")
    assert "command_term.cfg.ranges.ang_vel_z" in probe
    assert "sampled_yaw_abs_max" in probe
    assert "chord_fast_worst_yaw_m" in probe
    assert "chord_slow_straight_m" in probe
    # And it must re-check what is inherited through two levels.
    assert "V8.1 duty floor lost" in probe
    assert "V9.1 termination fix lost" in probe
    assert "terrain_levels curriculum missing" in probe


# ---------------------------------------------------------------------------
# The early check, which is the thing V9.1 got wrong
# ---------------------------------------------------------------------------


def _checker():
    spec = importlib.util.spec_from_file_location("ctp", "check_terrain_progress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checker_would_have_failed_v9_1_which_the_threshold_version_passed():
    """The regression test for my own checker.

    V9.1's series peaked at 1.584 (iter 500) and read 0.913 at iteration 1500.
    The threshold version asked only "is 0.913 >= 0.5?" and said PASS while the
    curriculum was collapsing.
    """
    ctp = _checker()
    v9_1_series = [(0, 0.987), (250, 1.003), (500, 1.584), (750, 1.470),
                   (1000, 1.317), (1250, 1.065), (1500, 0.913)]

    old_style_threshold_pass = v9_1_series[-1][1] >= 0.5
    assert old_style_threshold_pass, "this is exactly what the old check saw"

    result = ctp.verdict(v9_1_series, 1500, ctp.TERRAIN_LEVEL_MIN_EXPECTED,
                         ctp.TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK)
    assert result["passed"] is False
    assert result["declining"] is True
    assert result["peak_level"] == pytest.approx(1.584)
    assert result["peak_iteration"] == 500
    assert "peaked" in result["conclusion"]


def test_checker_passes_a_curriculum_that_climbs_and_holds():
    ctp = _checker()
    climbing = [(0, 0.9), (500, 1.6), (1000, 2.4), (1500, 3.1)]
    result = ctp.verdict(climbing, 1500, ctp.TERRAIN_LEVEL_MIN_EXPECTED,
                         ctp.TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK)
    assert result["passed"] is True
    assert result["declining"] is False
    assert result["still_in_v9_stalled_band"] is False


def test_checker_still_recognises_v9s_dead_flat_stall():
    ctp = _checker()
    stalled = [(0, 0.212), (500, 0.152), (1000, 0.164), (1500, 0.180)]
    result = ctp.verdict(stalled, 1500, ctp.TERRAIN_LEVEL_MIN_EXPECTED,
                         ctp.TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK)
    assert result["passed"] is False
    assert result["still_in_v9_stalled_band"] is True
    assert "did not affect promotion" in result["conclusion"]


def test_checker_ignores_the_startup_transient_when_finding_the_peak():
    """rsl_rl logs a high level at iteration 0 that is not curriculum behaviour.

    ``init_at_random_ep_len=True`` truncates most first episodes, and both V9.1
    and V9.2 logged exactly 0.987 at iteration 0. Counting that as the peak would
    fail a run that climbs to a healthy plateau below it.
    """
    ctp = _checker()
    # Startup spike, then a genuine climb to a stable plateau above the bar.
    series = [(0, 0.987), (71, 0.098), (500, 1.30), (1000, 1.62), (1500, 1.58)]
    result = ctp.verdict(series, 1500, 1.5, 0.25)
    assert result["peak_iteration"] == 1000
    assert result["peak_level"] == pytest.approx(1.62)
    assert result["passed"] is True

    # With no warmup the transient is still ignorable here, but the guard must be
    # explicit and reported so a future reader knows the peak window.
    assert result["warmup_iterations"] == ctp.TERRAIN_LEVEL_WARMUP_ITER
    assert ctp.TERRAIN_LEVEL_WARMUP_ITER > 0

    # And it must not swallow a real early peak that lies past the warmup.
    real_peak = ctp.verdict([(500, 1.584), (1500, 0.913)], 1500, 1.5, 0.25)
    assert real_peak["peak_iteration"] == 500
    assert real_peak["passed"] is False


def test_checker_falls_back_when_every_sample_is_inside_warmup():
    """A short log must still produce a verdict rather than crash on max([])."""
    ctp = _checker()
    result = ctp.verdict([(1500, 1.9)], 1500, 1.5, 0.25, warmup=100000)
    assert result is not None
    assert result["peak_iteration"] == 1500


def test_checker_waits_until_the_check_iteration():
    ctp = _checker()
    assert ctp.verdict([(400, 1.9)], 1500, 1.5, 0.25) is None
    assert ctp.verdict([], 1500, 1.5, 0.25) is None


def test_checker_pairs_each_level_with_the_iteration_above_it(tmp_path):
    ctp = _checker()
    log = tmp_path / "train.log"
    log.write_text(
        "                     Learning iteration 499/8000\n"
        "         Curriculum/terrain_levels: 1.5840\n"
        "                     Learning iteration 1500/8000\n"
        "               Episode_Reward/gait: 7.5960\n"
        "         Curriculum/terrain_levels: 0.9130\n",
        encoding="utf-8",
    )
    series = ctp.read_series(log)
    assert series == [(499, pytest.approx(1.584)), (1500, pytest.approx(0.913))]


def test_checker_constants_track_the_config_and_need_no_simulator():
    ctp_source = _source("check_terrain_progress.py")
    assert "from v9_2_env_cfg import" not in ctp_source
    assert "import isaaclab" not in ctp_source
    for name in ("TERRAIN_LEVEL_CHECK_ITER", "TERRAIN_LEVEL_MIN_EXPECTED",
                 "TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK"):
        assert _const(name, "check_terrain_progress.py") == _const(name)
    # The warmup window lives only in the checker; it describes log-reading, not
    # the environment, so it has no counterpart in the config.
    assert _const("TERRAIN_LEVEL_WARMUP_ITER", "check_terrain_progress.py") == 250
    # The bar must sit above where V9.1 plateaued, or it re-passes that failure.
    assert _const("TERRAIN_LEVEL_MIN_EXPECTED") > 0.913


def test_queue_and_tooling_target_v9_2():
    queue = _source("queue_v9_2_pilot.sh")
    assert "probe_v9_2.py" in queue and "train_v9_2.py" in queue
    assert "logs/rsl_rl/guide_v9_2_terrain" in queue
    assert queue.count("--env v9_2") >= 6
    assert "check_terrain_progress.py" in queue
    for name in ("evaluate_v9.py", "play_v9_video.py"):
        source = _source(name)
        # Newer versions extend the choice list; V9.2 must remain selectable so
        # its checkpoints stay evaluable under the config they trained with.
        assert '"v9_2"' in source
        assert "from v9_2_env_cfg import GuideTerrainTurnBandEnvCfg" in source


@pytest.mark.parametrize(
    "path", ["v9_2_env_cfg.py", "v9_2_ppo_cfg.py", "train_v9_2.py", "probe_v9_2.py"]
)
def test_v9_2_sources_parse(path):
    ast.parse(_source(path))
