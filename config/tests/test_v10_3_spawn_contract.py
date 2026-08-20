"""Contract tests for V10.3: the robot starts above the floor, not inside it.

Every run on this robot spawned 0.81 m underground. ``spawn_trace_probe.py``
measured root z = -0.3700 before any step -- exactly ``init_state.pos.z`` -- with
2.432x body weight of contact force on step 1 and a 0.815 m rise before settling
at +0.4451.

The cause was a pair of sign-inverted constants, both annotated as measured:
``FEET_ABOVE_ROOT_FRAME_M = 0.42`` (the feet are 0.445 m *below* the root frame)
and ``SETTLED_ROOT_FRAME_Z_M = -0.4192``. Nothing caught it because a flat
GroundPlane ejects the robot and it settles correctly, and every existing check
measured the settled state.
"""

import ast
from pathlib import Path

import pytest


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _const(name, module):
    tree = ast.parse(_source(module))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {module}")


# -- the constants themselves ------------------------------------------------

def _assigned_names(module):
    """Every module-level assignment target. Comments and docstrings excluded.

    These tests originally matched raw source text and tripped on the comment
    blocks that document what the old constants were -- the same brittleness that
    let a formatting change redden test_v8_recipe_contract for a whole version.
    Assert on structure instead.
    """
    tree = ast.parse(_source(module))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        # An annotated assignment is still an assignment: without this a typed
        # ``FEET_ABOVE_ROOT_FRAME_M: float = 0.42`` would slip past every check here.
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _spawn_expression(module):
    """The right-hand side of ``SPAWN_ROOT_FRAME_Z_M = ...`` as an AST node."""
    tree = ast.parse(_source(module))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "SPAWN_ROOT_FRAME_Z_M":
                return node.value
    raise AssertionError(f"SPAWN_ROOT_FRAME_Z_M not assigned in {module}")


def test_the_feet_are_recorded_as_below_the_root_frame():
    """The measurement: settled root z is positive with the feet on the ground."""
    names = _assigned_names("guide_robot_v2_cfg.py")
    assert "FEET_BELOW_ROOT_FRAME_M" in names
    # The inverted name must be gone as *code*, not merely unused -- other
    # modules imported it. Mentioning it in a comment is fine and expected.
    assert "FEET_ABOVE_ROOT_FRAME_M" not in names
    below = _const("FEET_BELOW_ROOT_FRAME_M", "guide_robot_v2_cfg.py")
    settled = _const("SETTLED_ROOT_FRAME_Z_M", "guide_robot_v2_cfg.py")
    assert below == pytest.approx(0.445, abs=1e-6)
    # Settled root z must be positive: the trunk sits above the ground.
    assert settled > 0.0
    assert settled == pytest.approx(below, abs=0.02)


def test_spawn_is_derived_by_addition_not_subtraction():
    """`clearance - 0.42` is what buried the robot; the sign is the whole bug."""
    for module in ("guide_robot_v2_cfg.py", "v10_flat_env_cfg.py"):
        expr = _spawn_expression(module)
        assert isinstance(expr, ast.BinOp), f"{module}: spawn is not a binary expression"
        assert isinstance(expr.op, ast.Add), f"{module}: spawn still subtracts"
        operands = {n.id for n in (expr.left, expr.right) if isinstance(n, ast.Name)}
        assert operands == {"FEET_SPAWN_CLEARANCE_M", "FEET_BELOW_ROOT_FRAME_M"}, operands


@pytest.mark.parametrize(
    ("module", "expected"),
    (("v10_flat_env_cfg.py", 0.05 + 0.445), ("guide_robot_v2_cfg.py", 0.32 + 0.445)),
)
def test_every_spawn_height_is_above_the_floor(module, expected):
    clearance = _const("FEET_SPAWN_CLEARANCE_M", module)
    below = _const("FEET_BELOW_ROOT_FRAME_M", "guide_robot_v2_cfg.py")
    spawn = clearance + below
    assert spawn == pytest.approx(expected, abs=1e-9)
    # The property that matters: the feet start above the ground, not under it.
    assert spawn - below > 0.0, "feet would start at or below the floor"
    # And the old arithmetic must be demonstrably wrong, so this test can fail.
    assert clearance - below < 0.0, "the pre-V10.3 formula put the feet underground"


def _imported_names(module):
    """Every name pulled in by an import statement. Comments excluded."""
    tree = ast.parse(_source(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def test_no_module_still_imports_the_inverted_constant():
    for module in ("v10_flat_env_cfg.py", "v10_env_cfg.py"):
        assert "FEET_ABOVE_ROOT_FRAME_M" not in _imported_names(module)


def test_the_dead_settled_import_is_gone_from_the_terrain_config():
    """v10_env_cfg imported SETTLED_ROOT_FRAME_Z_M and never used it.

    Same shape as probe_v10_1_flat.py assigning `duty_col` and never using it:
    an unused reference reads as coverage that is not there.
    """
    assert "SETTLED_ROOT_FRAME_Z_M" not in _imported_names("v10_env_cfg.py")


# -- the probe check that would have caught it -------------------------------

def test_the_probe_reads_the_spawn_state_before_stepping():
    """Every previous check measured the settled state, which looked healthy."""
    probe = _source("probe_v10_3_flat.py")
    assert "root_z_before_any_step" in probe
    assert "robot spawns at z=" in probe
    assert "this is the V10.1/V10.2 defect" in probe
    # Reading it is not enough; it has to be compared to something derived.
    assert "V10_3_FEET_BELOW_ROOT_FRAME_M + V10_3_FEET_SPAWN_CLEARANCE_M" in probe


def test_the_probe_bounds_the_first_step_contact_impulse():
    """The defect showed 2.432x body weight on step 1; a clean spawn touches nothing."""
    probe = _source("probe_v10_3_flat.py")
    assert "support_on_first_step" in probe
    assert "the robot is starting inside the floor" in probe
    assert 'report["support_on_first_step"] > 0.5' in probe


def test_the_probe_keeps_every_check_it_inherited():
    """V10.2's probe was itself a repair of checks dropped during a port."""
    probe = _source("probe_v10_3_flat.py")
    for inherited in (
        "term._duty[:] = 1.0",
        "term._duty[:, 0] = 0.0",
        "duty floor did not fire on an abandoned foot",
        "v5_12_mdp._moving_mask(base,",
        "fore_aft_axis_measured",
        "support_fraction",
        "tilt_deg_mean",
        # The rest of V10.2's contract, which an earlier version of this test
        # silently omitted while claiming to cover "every check".
        "obs_width",
        "action_scale",
        "contact_bodies",
        "V8 weights drifted",
        "duty_floor weight",
        "termination_terms",
        "mass_one_env_kg",
        "cmd_lin_vel_x",
        "the duty floor would be exempt at the low end",
    ):
        assert inherited in probe, f"{inherited} lost in the port to V10.3"


# -- the evaluator warmup flag -----------------------------------------------

def test_warmup_defaults_to_zero_so_archived_numbers_are_unchanged():
    evaluator = _source("evaluate_v8.py")
    assert 'parser.add_argument("--warmup-seconds", type=float, default=0.0)' in evaluator
    assert "warmup_steps = int(args_cli.warmup_seconds / base_env.step_dt)" in evaluator
    assert "if _step >= warmup_steps:" in evaluator
    # A scored JSON must say how it was scored.
    assert '"warmup_seconds": args_cli.warmup_seconds,' in evaluator


def test_warmup_cannot_consume_the_whole_rollout():
    assert "leaves no rollout" in _source("evaluate_v8.py")


def test_survival_still_accrues_from_step_zero():
    """A fall during the warmup is a real fall and must not be discarded."""
    evaluator = _source("evaluate_v8.py")
    body = evaluator.split("for _step in range(steps):")[1]
    alive_line = "alive &= ~dones.bool()"
    assert alive_line in body
    # The alive update must sit outside the warmup guard.
    guard = body.index("if _step >= warmup_steps:")
    assert body.index(alive_line) > guard
    indent = body.split(alive_line)[0].rsplit("\n", 1)[-1]
    assert len(indent) == 16, f"alive update is nested at indent {len(indent)}"


# -- version scaffolding -----------------------------------------------------

def test_v10_3_changes_the_spawn_and_nothing_in_its_environment():
    source = _source("v10_3_flat_env_cfg.py")
    assert "class GuideV2FlatSpawnFixedEnvCfg(GuideV2FlatDutyLiveEnvCfg)" in source
    assert "V10_3_FLAT_REWARD_WEIGHTS = dict(V10_2_FLAT_REWARD_WEIGHTS)" in source
    assert "__post_init__" not in source
    for untouched in ("self.rewards", "self.commands", "self.terminations",
                      "self.scene", "self.observations", "self.actions"):
        assert untouched not in source


def test_v10_3_agent_config_is_v10_2_deep_copied():
    ppo = _source("v10_3_flat_ppo_cfg.py")
    assert "copy.deepcopy(V10_2_FLAT_PPO_CFG)" in ppo
    assert 'V10_3_FLAT_PPO_CFG.experiment_name = "guide_v10_3_flat_robot_v2"' in ppo
    for forbidden in ("init_noise_std", "actor_hidden_dims", "num_steps_per_env",
                      "learning_rate", "max_iterations", "seed"):
        assert forbidden not in ppo


def test_v10_3_tooling_targets_v10_3():
    queue = _source("queue_v10_3_flat_pilot.sh")
    assert "probe_v10_3_flat.py" in queue and "train_v10_3_flat.py" in queue
    assert "logs/rsl_rl/guide_v10_3_flat_robot_v2" in queue
    assert queue.count("--env v10_3_flat") >= 4
    assert queue.count("--forward-axis +y") >= 4
    assert "v10_2" not in queue


def test_older_checkpoints_stay_evaluable():
    for name in ("evaluate_v8.py", "play_v8_video.py"):
        source = _source(name)
        for older in ("v10_flat", "v10_1_flat", "v10_2_flat", "v10_3_flat"):
            assert older in source


@pytest.mark.parametrize(
    "path",
    ["v10_3_flat_env_cfg.py", "v10_3_flat_ppo_cfg.py", "train_v10_3_flat.py",
     "probe_v10_3_flat.py", "spawn_trace_probe.py"],
)
def test_v10_3_sources_parse(path):
    ast.parse(_source(path))
