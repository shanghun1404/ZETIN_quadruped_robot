"""Contract tests for V10.4: V10.3's gait on rough terrain, blind.

Two things have already been measured and must not be re-litigated by accident:

* **The curriculum vanishes if it is only declared.** V8's ``__post_init__`` sets
  ``self.curriculum = None`` and runs *after* a class attribute binds, so V9
  shipped a config whose source clearly declared ``terrain_levels`` while the
  manager held nothing. Its probe caught it before launch.
* **The inherited yaw band makes promotion impossible.** ``terrain_levels_vel``
  promotes on a straight-line displacement past ``size[0] / 2`` = 4.0 m. At V8's
  inherited ±0.6 the 20 s chord is 0.466 m; at V9.2's ±0.2 it is 4.546 m.
"""

import ast
import math
from pathlib import Path

import pytest


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _const(name, module="v10_4_terrain_env_cfg.py"):
    """Resolve a module-level constant, including ones defined from other constants.

    ``ast.literal_eval`` raises on ``PROMOTION_DISTANCE_M = TERRAIN_TILE_SIZE_M / 2.0``
    because a BinOp is not a literal. Deriving one constant from another is the
    behaviour these tests want to protect -- it is what keeps the promotion
    distance tied to the tile size instead of being a hand-typed number -- so the
    resolver handles it rather than the config being flattened to suit the test.
    """
    tree = ast.parse(_source(module))
    assigns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            assigns[node.targets[0].id] = node.value

    def resolve(expr, depth=0):
        if depth > 8:
            raise AssertionError(f"constant resolution for {name} did not terminate")
        if isinstance(expr, ast.Name):
            if expr.id not in assigns:
                raise AssertionError(f"{expr.id} not defined in {module}")
            return resolve(assigns[expr.id], depth + 1)
        if isinstance(expr, ast.BinOp):
            left, right = resolve(expr.left, depth + 1), resolve(expr.right, depth + 1)
            op = type(expr.op)
            if op is ast.Add:
                return left + right
            if op is ast.Sub:
                return left - right
            if op is ast.Mult:
                return left * right
            if op is ast.Div:
                return left / right
            raise AssertionError(f"unsupported operator {op.__name__} in {name}")
        return ast.literal_eval(expr)

    if name not in assigns:
        raise AssertionError(f"{name} not found in {module}")
    return resolve(assigns[name])


def _chord(v, w, t):
    return abs((2.0 * v / w) * math.sin(w * t / 2.0)) if w else v * t


# -- the yaw band is arithmetic, not preference -------------------------------

def test_the_inherited_yaw_band_makes_promotion_impossible():
    """The reason V10.4 is allowed a second change: without it the run is dead."""
    inherited = _const("V8_YAW_RATE_RANGE", "v8_env_cfg.py")
    assert inherited == (-0.6, 0.6)
    chord = _chord(0.5, 0.6, 20.0)
    assert chord < 4.0, chord
    assert chord == pytest.approx(0.466, abs=0.01)


def test_the_chosen_yaw_band_clears_the_promotion_distance():
    band = _const("V10_4_YAW_RATE_RANGE")
    assert band == (-0.2, 0.2)
    chord = _chord(0.5, 0.2, 20.0)
    assert chord > _const("PROMOTION_DISTANCE_M")
    assert chord == pytest.approx(4.546, abs=0.01)


def test_promotion_distance_matches_the_curriculum_rule():
    """terrain_levels_vel compares against size[0] / 2, not a tuned number."""
    assert _const("PROMOTION_DISTANCE_M") == _const("TERRAIN_TILE_SIZE_M") / 2.0


def test_the_helper_agrees_with_the_closed_form():
    source = _source("v10_4_terrain_env_cfg.py")
    assert "def displacement_chord_m" in source
    assert "math.sin(yaw_rate_rad_s * seconds / 2.0)" in source
    # A zero yaw rate is a straight line, not a division by zero.
    assert "if yaw_rate_rad_s == 0.0:" in source


# -- the curriculum trap ------------------------------------------------------

def test_the_curriculum_is_restored_inside_post_init_not_only_declared():
    """Declaring it as a class attribute alone is what silently killed V9."""
    source = _source("v10_4_terrain_env_cfg.py")
    assert "curriculum: V10_4CurriculumCfg = V10_4CurriculumCfg()" in source
    body = source.split("def __post_init__")[1]
    assert "self.curriculum = V10_4CurriculumCfg()" in body
    # And it must come after super(), which is what nulls it.
    assert body.index("super().__post_init__()") < body.index("self.curriculum =")


def test_the_probe_asserts_the_curriculum_is_actually_registered():
    probe = _source("probe_v10_4_terrain.py")
    assert "base.curriculum_manager.active_terms" in probe
    assert "terrain_levels curriculum missing" in probe
    assert "which is exactly what happened to V9" in probe


def test_the_probe_asserts_the_chord_before_training_starts():
    """A yaw band that cannot promote wastes the whole run; catch it in seconds."""
    probe = _source("probe_v10_4_terrain.py")
    assert "worst_case_chord_m" in probe
    assert "no environment could ever " in probe
    # The tile size must be read from the terrain, not assumed.
    assert "base.cfg.scene.terrain.terrain_generator.size[0]" in probe


# -- blind is the point -------------------------------------------------------

def test_the_scanner_is_off_and_the_observation_stays_v8_width():
    source = _source("v10_4_terrain_env_cfg.py")
    assert "self.scene.height_scanner = None" in source
    # No V9-style observation subclass: the policy sees exactly what V10.3 saw.
    assert "ObsTerm" not in source
    assert "height_scan" not in source.split('"""')[2]  # not in code, only prose

    probe = _source("probe_v10_4_terrain.py")
    assert "V10.4 is blind by design" in probe
    assert "height_scan observation present on a blind configuration" in probe
    assert 'report["obs_width"] != 48' in probe


def test_terrain_generation_is_on_and_starts_at_the_easiest_tile():
    source = _source("v10_4_terrain_env_cfg.py")
    assert 'self.scene.terrain.terrain_type = "generator"' in source
    assert "self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG" in source
    assert "self.scene.terrain.max_init_terrain_level = 0" in source
    assert "self.scene.terrain.terrain_generator.curriculum = True" in source


# -- the spawn, which is the open risk ---------------------------------------

def test_the_terrain_spawn_uses_the_corrected_sign():
    source = _source("v10_4_terrain_env_cfg.py")
    assert "TERRAIN_SPAWN_ROOT_FRAME_Z_M = FEET_SPAWN_CLEARANCE_M + FEET_BELOW_ROOT_FRAME_M" in source
    assert "- FEET_ABOVE_ROOT_FRAME_M" not in source


def test_the_probe_measures_spawn_relative_to_the_tile_origin():
    """World z is meaningless on terrain: each tile's origin sits at its surface."""
    probe = _source("probe_v10_4_terrain.py")
    assert "base.scene.env_origins[:, 2]" in probe
    assert "rel_z = robot_pre.data.root_pos_w[:, 2] - origins_z" in probe
    assert "root_z_before_any_step_min" in probe
    # And the first-step impulse bound that caught the underground spawn.
    assert 'report["support_on_first_step"] > 0.5' in probe


def test_the_probe_keeps_every_check_it_inherited():
    probe = _source("probe_v10_4_terrain.py")
    for inherited in (
        "term._duty[:] = 1.0",
        "term._duty[:, 0] = 0.0",
        "duty floor did not fire on an abandoned foot",
        "v5_12_mdp._moving_mask(base,",
        "fore_aft_axis_measured",
        "support_fraction",
        "tilt_deg_mean",
        "obs_width",
        "action_scale",
        "contact_bodies",
        "V8 weights drifted",
        "duty_floor weight",
        "termination_terms",
        "mass_one_env_kg",
        "cmd_lin_vel_x",
    ):
        assert inherited in probe, f"{inherited} lost in the port to V10.4"


# -- version scaffolding ------------------------------------------------------

def test_v10_4_inherits_v10_3_and_changes_only_ground_and_yaw():
    source = _source("v10_4_terrain_env_cfg.py")
    assert "class GuideV2BlindTerrainEnvCfg(GuideV2FlatSpawnFixedEnvCfg)" in source
    assert "V10_4_REWARD_WEIGHTS = dict(V10_3_FLAT_REWARD_WEIGHTS)" in source
    body = source.split("def __post_init__")[1]
    # The reward table, terminations, actions and observations must not be touched.
    for untouched in ("self.rewards", "self.terminations", "self.actions", "self.observations"):
        assert untouched not in body
    # Exactly one command range moves: the yaw band.
    assert body.count("self.commands.") == 1
    assert "ang_vel_z = V10_4_YAW_RATE_RANGE" in body


def test_v10_4_agent_config_is_v10_3_deep_copied():
    ppo = _source("v10_4_terrain_ppo_cfg.py")
    assert "copy.deepcopy(V10_3_FLAT_PPO_CFG)" in ppo
    assert 'V10_4_TERRAIN_PPO_CFG.experiment_name = "guide_v10_4_terrain_robot_v2"' in ppo
    for forbidden in ("init_noise_std", "actor_hidden_dims", "num_steps_per_env",
                      "learning_rate", "max_iterations", "seed"):
        assert forbidden not in ppo


def test_v10_4_tooling_targets_v10_4():
    queue = _source("queue_v10_4_terrain_pilot.sh")
    assert "probe_v10_4_terrain.py" in queue and "train_v10_4_terrain.py" in queue
    assert "logs/rsl_rl/guide_v10_4_terrain_robot_v2" in queue
    assert queue.count("--env v10_4_terrain") >= 4
    assert queue.count("--forward-axis +y") >= 4
    assert "v10_3" not in queue


def test_older_checkpoints_stay_evaluable():
    for name in ("evaluate_v8.py", "play_v8_video.py"):
        source = _source(name)
        for older in ("v10_flat", "v10_1_flat", "v10_2_flat", "v10_3_flat", "v10_4_terrain"):
            assert older in source


@pytest.mark.parametrize(
    "path",
    ["v10_4_terrain_env_cfg.py", "v10_4_terrain_ppo_cfg.py", "train_v10_4_terrain.py",
     "probe_v10_4_terrain.py"],
)
def test_v10_4_sources_parse(path):
    ast.parse(_source(path))
