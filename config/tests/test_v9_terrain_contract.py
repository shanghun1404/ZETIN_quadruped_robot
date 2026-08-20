import ast
from pathlib import Path


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _const(name, module="v9_env_cfg.py"):
    tree = ast.parse(_source(module))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == name:
                return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [e.id for e in node.targets[0].elts]
            if name in names:
                values = ast.literal_eval(node.value)
                return values[names.index(name)]
    raise AssertionError(f"{name} not found")


def test_v9_keeps_the_v8_1_reward_table_untouched():
    # V9 is a perception/terrain change; a simultaneous reward edit would make
    # any terrain failure unattributable.
    source = _source("v9_env_cfg.py")
    assert "V9_REWARD_WEIGHTS = dict(V8_1_REWARD_WEIGHTS)" in source
    assert "self.rewards" not in source


def test_v9_builds_on_the_duty_floored_gait_not_the_bare_spot_recipe():
    """Terrain must not be asked to repair a defect measured on flat ground.

    V8's 16/84 contact split is a flat-ground failure of the reward table. If V9
    inherited that table, a terrain result could not distinguish "the terrain is
    hard" from "the gait was already broken".
    """
    source = _source("v9_env_cfg.py")
    assert "class GuideTerrainEnvCfg(GuideDutyFloorEnvCfg)" in source
    assert "from v8_1_env_cfg import GuideDutyFloorEnvCfg, V8_1_REWARD_WEIGHTS" in source
    assert "GuideSpotRecipeEnvCfg" not in source

    # And the floor must survive the inheritance rather than being a dead name.
    # V8_1_REWARD_WEIGHTS spreads V8's dict, which ast.literal_eval cannot
    # evaluate, so assert on the structure that defines it.
    v8_1 = _source("v8_1_env_cfg.py")
    assert "**V8_REWARD_WEIGHTS" in v8_1
    assert '"duty_floor": DUTY_FLOOR_WEIGHT' in v8_1
    assert "self.rewards.duty_floor = RewardTermCfg(" in v8_1


def test_v9_restores_the_height_scanner_looking_forward():
    source = _source("v9_env_cfg.py")
    assert "self.scene.height_scanner = RayCasterCfg(" in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/base_link"' in source
    # Guide-forward is base -x, so the scan offset must be negative.
    assert _const("SCAN_OFFSET_X") == -0.8
    assert _const("SCAN_OFFSET_X") < 0.0
    assert _const("SCAN_RESOLUTION") == 0.1
    assert _const("SCAN_SIZE") == [1.6, 1.0]


def test_v9_scan_footprint_matches_the_stage2_camera_target():
    # Stage 2 distils a forward camera onto exactly this scan, so the point
    # count is part of that contract. SCAN_POINTS is a product expression, so
    # check the factors and the arithmetic rather than literal-evaluating it.
    rows, cols = _const("SCAN_ROWS"), _const("SCAN_COLS")
    assert (rows, cols) == (17, 11)
    assert rows * cols == 187
    assert "SCAN_POINTS = SCAN_ROWS * SCAN_COLS" in _source("v9_env_cfg.py")

    # The footprint must follow from the grid: size / resolution + 1 per axis.
    size = _const("SCAN_SIZE")
    resolution = _const("SCAN_RESOLUTION")
    assert round(size[0] / resolution) + 1 == rows
    assert round(size[1] / resolution) + 1 == cols


def test_v9_enables_generated_terrain_with_a_curriculum():
    source = _source("v9_env_cfg.py")
    assert 'self.scene.terrain.terrain_type = "generator"' in source
    assert "self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG" in source
    # Start on the easiest tile so the V8 gait is re-learned before slopes.
    assert "self.scene.terrain.max_init_terrain_level = 0" in source
    assert "terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)" in source


def test_v9_curriculum_survives_v8s_flat_ground_post_init():
    """Declaring the curriculum is not enough -- V8's __post_init__ deletes it.

    ``GuideSpotRecipeEnvCfg.__post_init__`` sets ``self.curriculum = None`` for
    flat ground and runs before V9's body, so a V9 that only declares the class
    attribute trains with no terrain promotion whatsoever and never leaves the
    easiest tile. probe_v9 caught exactly that, with an empty ``curriculum_terms``
    against a source that plainly declared the term.
    """
    source = _source("v9_env_cfg.py")
    assert "self.curriculum = None" in _source("v8_env_cfg.py"), "the hazard this guards"
    body = source.split("def __post_init__(self):")[1]
    assert "self.curriculum = V9CurriculumCfg()" in body
    # And it must be restored after the parent has run, not before.
    assert body.index("super().__post_init__()") < body.index("self.curriculum = V9CurriculumCfg()")
    # The probe must keep checking the resolved terms, not the declaration.
    assert "curriculum_terms" in _source("probe_v9.py")


def test_v9_observation_adds_the_scan_on_top_of_v8():
    source = _source("v9_env_cfg.py")
    assert "class PolicyCfg(V8ObservationsCfg.PolicyCfg)" in source
    assert "height_scan = ObsTerm(" in source
    assert "func=mdp.height_scan" in source
    assert "clip=(-1.0, 1.0)" in source


def test_v9_agent_config_is_unchanged_from_v8_apart_from_naming():
    v8 = _source("v8_ppo_cfg.py")
    v9 = _source("v9_ppo_cfg.py")
    for literal in (
        "init_noise_std=1.0",
        "entropy_coef=0.0025",
        "num_steps_per_env=24",
        "actor_hidden_dims=[512, 256, 128]",
        "max_iterations=20000",
    ):
        assert literal in v8 and literal in v9
    assert 'experiment_name="guide_v9_terrain"' in v9


def test_v9_probe_checks_the_scan_is_alive_and_the_right_width():
    probe = _source("probe_v9.py")
    assert "expected_obs_width" in probe
    assert "48 + SCAN_POINTS" in probe
    assert "scan_std_across_envs" in probe
    assert "terrain_levels" in probe


def test_stairs_scenario_uses_the_ascending_terrain_variant():
    """A pyramid spawns the robot on top (descending); only the inverted one climbs.

    The terrain origin is ``+(num_steps+1)*step_height`` for a pyramid and
    ``-(num_steps+1)*step_height`` for an inverted pyramid, so the robot starts
    in the pit and walks *up* only in the inverted case.
    """
    source = _source("v9_env_cfg.py")
    assert "MeshInvertedPyramidStairsTerrainCfg" in source
    # The descending variant must not be what the climb test measures.
    assert "STAIRS_UP_EVAL_TERRAIN_CFG" in source
    stairs_up_block = source.split("STAIRS_UP_EVAL_TERRAIN_CFG")[1].split(
        "STAIRS_DOWN_EVAL_TERRAIN_CFG"
    )[0]
    assert "MeshPyramidStairsTerrainCfg" not in stairs_up_block
    stairs_down_block = source.split("STAIRS_DOWN_EVAL_TERRAIN_CFG")[1].split(
        "@configclass"
    )[0]
    assert "MeshPyramidStairsTerrainCfg" in stairs_down_block
    assert "MeshInvertedPyramidStairsTerrainCfg" not in stairs_down_block


def test_stairs_terrain_brackets_a_real_building_stair():
    low, high = _const("STAIRS_STEP_HEIGHT_RANGE")
    # Real indoor stairs are about 0.15-0.18 m; the range must cover that.
    assert low == 0.05 and high == 0.20
    assert low < 0.15 < high
    assert _const("STAIRS_STEP_WIDTH") == 0.30
    assert _const("STAIRS_NUM_LEVELS") == 8


def test_stairs_terrain_is_stairs_only_and_graded_by_row():
    source = _source("v9_env_cfg.py")
    block = source.split("STAIRS_UP_EVAL_TERRAIN_CFG = TerrainGeneratorCfg(")[1].split("\n)")[0]
    assert "proportion=1.0" in block          # stairs only, no other sub-terrain
    assert "curriculum=True" in block          # row index == difficulty
    assert "num_rows=STAIRS_NUM_LEVELS" in block


def test_evaluator_exposes_the_stairs_scenario_with_its_own_gates():
    ev = _source("evaluate_v9.py")
    assert 'choices=("rough", "stairs_up")' in ev
    assert "env_cfg.scene.terrain.terrain_generator = STAIRS_UP_EVAL_TERRAIN_CFG" in ev
    # Levels must be frozen, otherwise per-level success is meaningless.
    assert "env_cfg.curriculum = None" in ev
    assert "max_init_terrain_level = STAIRS_NUM_LEVELS - 1" in ev
    for gate in ("stairs_survival", "stairs_climb_success", "stairs_clears_a_real_step"):
        assert gate in ev
    # Success must scale with the level's own step height, not a flat threshold,
    # AND require travel up the structure. The arithmetic moved into a
    # simulator-free helper so it can be executed here instead of grepped for.
    assert "from v9_eval_metrics import climbed_stairs" in ev
    assert "climbed = climbed_stairs(" in ev
    assert "hardest_level_cleared" in ev

    import importlib.util

    spec = importlib.util.spec_from_file_location("v9m", "v9_eval_metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)

    class _Flag:
        """Minimal stand-in so the helper can be exercised without torch."""

        def __init__(self, value):
            self.value = value

        def __ge__(self, other):
            return _Flag(self.value >= other)

        def __and__(self, other):
            return bool(self.value) and bool(other.value)

        def __bool__(self):
            return bool(self.value)

    step_h, approach = 0.15, 2.1
    # Enough height but no travel: a hop in the starting pit is not a climb.
    assert not metrics.climbed_stairs(_Flag(0.40), _Flag(0.2), step_h, approach)
    # Travel but not enough height: crossing the platform is not a climb either.
    assert not metrics.climbed_stairs(_Flag(0.10), _Flag(3.0), step_h, approach)
    # Both: a real climb.
    assert metrics.climbed_stairs(_Flag(0.40), _Flag(3.0), step_h, approach)
    # And the height bar must track the level's own step height.
    assert not metrics.climbed_stairs(_Flag(0.40), _Flag(3.0), 0.25, approach)


def test_phase_correlation_reports_undefined_rather_than_a_passing_zero():
    """A constant joint signal has no correlation; reporting 0.0 passed every gate.

    Clamping the Pearson denominator turned an undefined correlation into 0.0,
    which satisfies both same-side gates (<= 0.30) and both diagonal gates
    (>= 0.0). After V9.2 converged on a near-stationary policy that is not a
    hypothetical failure mode.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("v9m", "v9_eval_metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)

    assert metrics.pearson_correlation_or_none(0.0, 0.0, 1.0) is None
    assert metrics.pearson_correlation_or_none(0.0, 1.0, 0.0) is None
    assert metrics.pearson_correlation_or_none(0.5, 1.0, 1.0) == 0.5

    # And the gates must refuse to pass on a missing value.
    ev = _source("evaluate_v9.py")
    assert "pearson_correlation_or_none" in ev
    assert "is not None" in ev


def test_queue_runs_the_stairs_gate_and_records_stairs_video():
    q = _source("queue_v9_pilot.sh")
    assert "--scenario stairs_up" in q
    assert "v9_stairs_gate.json" in q
    assert q.count("--scenario stairs_up") >= 3   # one gate + two videos
    # The stairs run needs longer than the flat one to actually reach the steps.
    assert "--seconds 12" in q


def test_stairs_video_is_pinned_to_a_realistic_step_not_the_easiest_row():
    video = _source("play_v9_video.py")
    assert 'choices=("rough", "stairs_up")' in video
    assert "max_init_terrain_level = STAIRS_NUM_LEVELS - 3" in video
    # The scenario must appear in the filename so recordings are not confused.
    assert "{args_cli.scenario}" in video
