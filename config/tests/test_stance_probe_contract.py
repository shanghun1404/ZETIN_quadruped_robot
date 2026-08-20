import ast
from pathlib import Path

import pytest


PROBE_PATH = Path("probe_stance_calibration.py")


def _probe_source() -> str:
    return PROBE_PATH.read_text()


def _probe_tree(source: str) -> ast.Module:
    return ast.parse(source)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _assignments(node: ast.AST) -> dict[str, str]:
    result = {}
    for child in ast.walk(node):
        if isinstance(child, (ast.Assign, ast.AnnAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            value = child.value
            for target in targets:
                result[ast.unparse(target)] = ast.unparse(value)
    return result


def _calls(node: ast.AST) -> list[ast.Call]:
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def _attribute_step_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in _calls(node)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "step"
    ]


def _assert_no_runtime_manager_or_reset_calls(node: ast.AST) -> None:
    forbidden_paths = (
        ".reward_manager.",
        ".observation_manager.",
        ".termination_manager.",
        ".event_manager.",
    )
    for call in _calls(node):
        path = ast.unparse(call.func)
        assert path != "gym_env.step"
        assert not any(fragment in path for fragment in forbidden_paths)
        assert not path.endswith((".reset", "._reset_idx"))
        assert ".reset_" not in path


def _assert_sampling_loop(loop: ast.For, constants: dict[str, str]) -> None:
    assert isinstance(loop.iter, ast.Call)
    assert ast.unparse(loop.iter.func) == "range"
    assert len(loop.iter.args) == 1
    bound = ast.unparse(loop.iter.args[0])
    assert constants[bound] == "50"

    control_calls = [call for call in _calls(loop) if ast.unparse(call.func) == "control_step"]
    assert len(control_calls) == 1
    assert ast.unparse(control_calls[0]) == "control_step(base_env, zero_actions)"
    assert _attribute_step_calls(loop) == []
    _assert_no_runtime_manager_or_reset_calls(loop)

    accumulators = [
        (ast.unparse(child.target), ast.unparse(child.value))
        for child in ast.walk(loop)
        if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.BitOr)
    ]
    expected = {
        "body_collision": "bool(torch.any(body_forces > BODY_CONTACT_THRESHOLD_N).item())",
        "low_height": "bool(torch.any(height < LOW_HEIGHT_THRESHOLD_M).item())",
    }
    for target, value in expected.items():
        assert [expression for name, expression in accumulators if name == target] == [value]


def _validate_runtime_and_candidates(source: str) -> None:
    tree = _probe_tree(source)
    assignments = _assignments(tree)
    assert assignments["NUM_ENVS"] == "64"
    assert assignments["SETTLE_STEPS"] == "50"
    assert assignments["MEASUREMENT_STEPS"] == "50"
    assert assignments["CONTROL_STEP_S"] == "0.02"
    assert assignments["FOOT_CONTACT_THRESHOLD_N"] == "5.0"
    assert assignments["BODY_CONTACT_THRESHOLD_N"] == "1.0"
    assert assignments["LOW_HEIGHT_THRESHOLD_M"] == "0.15"


def _validate_environment_lifecycle(source: str) -> None:
    tree = _probe_tree(source)
    collector = _function(tree, "collect_candidate")
    statements = [ast.unparse(statement) for statement in collector.body]
    make_index = next(index for index, statement in enumerate(statements) if "gym.make(" in statement)
    before_make = "\n".join(statements[:make_index])
    for term in ("time_out", "base_contact", "low_height", "support_loss"):
        assert f"env_cfg.terminations.{term} = None" in before_make
    assert "env_cfg.scene.robot.init_state.pos = (0.0, 0.0, candidate.spawn_height_m)" in before_make
    assert "env_cfg.scene.robot.init_state.joint_pos['.*_HFE'] = candidate.hfe_rad" in before_make
    assert "env_cfg.scene.robot.init_state.joint_pos['.*_KFE'] = candidate.kfe_rad" in before_make
    assert "env_cfg.events.reset_base = None" not in before_make
    assert "env_cfg.events.reset_robot_joints = None" not in before_make
    assert (
        "env_cfg.events.reset_base.params['pose_range'] = deterministic_base_ranges"
        in before_make
    )
    assert (
        "env_cfg.events.reset_base.params['velocity_range'] = deterministic_base_ranges"
        in before_make
    )
    assert (
        "env_cfg.events.reset_robot_joints.params['position_range'] = (1.0, 1.0)"
        in before_make
    )
    assert (
        "env_cfg.events.reset_robot_joints.params['velocity_range'] = (0.0, 0.0)"
        in before_make
    )

    make_calls = [
        call for call in ast.walk(collector) if isinstance(call, ast.Call) and ast.unparse(call.func) == "gym.make"
    ]
    assert len(make_calls) == 1
    cleanup = next(node for node in collector.body if isinstance(node, ast.Try))
    assert ast.unparse(cleanup.finalbody[-1]) == "gym_env.close()"


def _validate_sampling_and_contacts(source: str) -> None:
    tree = _probe_tree(source)
    collector = _function(tree, "collect_candidate")
    assignments = _assignments(collector)

    assert assignments["ordered_feet"] == "['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']"
    assert assignments["ordered_sensor"] == (
        "SceneEntityCfg('contact_forces', body_names=ordered_feet, preserve_order=True)"
    )
    assert assignments["body_sensor"] == (
        "SceneEntityCfg('contact_forces', body_names='base_link|.*_hip')"
    )
    assert assignments["zero_actions"] == (
        "torch.zeros((NUM_ENVS, base_env.action_manager.total_action_dim), device=base_env.device)"
    )
    collector_source = ast.unparse(collector)
    assert "smooth_mdp.foot_contact_flags(base_env, FOOT_CONTACT_THRESHOLD_N, ordered_sensor)" in collector_source

    sampling_loops = [loop for loop in ast.walk(collector) if isinstance(loop, ast.For)]
    assert len(sampling_loops) == 2
    constants = _assignments(tree)
    loop_bounds = [ast.unparse(loop.iter.args[0]) for loop in sampling_loops]
    assert sorted(loop_bounds) == ["MEASUREMENT_STEPS", "SETTLE_STEPS"]
    for loop in sampling_loops:
        _assert_sampling_loop(loop, constants)

    collector_control_calls = [
        call for call in _calls(collector) if ast.unparse(call.func) == "control_step"
    ]
    assert len(collector_control_calls) == 2
    assert "control_step(base_env, zero_actions)" in source
    assert "gym_env.step(zero_actions)" not in source


def _validate_low_level_step(source: str) -> None:
    tree = _probe_tree(source)
    step = _function(tree, "control_step")
    step_source = ast.unparse(step)
    ordered_calls = (
        "base_env.action_manager.process_action(zero_actions.to(base_env.device))",
        "base_env.action_manager.apply_action()",
        "base_env.scene.write_data_to_sim()",
        "base_env.sim.step(render=False)",
        "base_env.scene.update(dt=base_env.physics_dt)",
    )
    positions = [step_source.index(call) for call in ordered_calls]
    assert positions == sorted(positions)
    assert "for _ in range(base_env.cfg.decimation):" in step_source
    assert "base_env._sim_step_counter += 1" in step_source
    assert "base_env.episode_length_buf += 1" in step_source
    assert "base_env.common_step_counter += 1" in step_source
    assert [ast.unparse(call) for call in _attribute_step_calls(step)] == [
        "base_env.sim.step(render=False)"
    ]
    _assert_no_runtime_manager_or_reset_calls(step)


def _validate_violations_and_evidence(source: str) -> None:
    tree = _probe_tree(source)
    collector_source = ast.unparse(_function(tree, "collect_candidate"))
    for literal in (
        "body_collision |= bool(torch.any(body_forces > BODY_CONTACT_THRESHOLD_N).item())",
        "low_height |= bool(torch.any(height < LOW_HEIGHT_THRESHOLD_M).item())",
        "height_min = torch.minimum(height_min, height)",
        "height_max = torch.maximum(height_max, height)",
        "sum_height += height.sum().item()",
        "sum_vertical_velocity_sq += robot.data.root_lin_vel_b[:, 2].square().sum().item()",
        "support_samples += (contact_count >= 2.0).sum().item()",
        "torch.quantile(height_max - height_min, 0.95).item()",
    ):
        assert literal in collector_source

    assert "StanceMetrics(" in collector_source
    assert "select_stance_candidate(metrics)" in source
    assert "passes_stance(candidate)" in source

    payload = next(
        node
        for node in ast.walk(_function(tree, "write_evidence"))
        if isinstance(node, ast.Dict)
        and {ast.literal_eval(key) for key in node.keys if key is not None}
        == {"target_height_m", "candidates", "selected_candidate"}
    )
    assert payload is not None
    evidence_keys = {
        "half_angle_rad",
        "hfe_rad",
        "kfe_rad",
        "spawn_height_m",
        "mean_height_m",
        "support_fraction",
        "rms_vertical_velocity_m_s",
        "p95_height_range_m",
        "body_collision",
        "low_height",
        "passed",
    }
    evidence_dict = next(
        node
        for node in ast.walk(_function(tree, "metrics_evidence"))
        if isinstance(node, ast.Dict)
    )
    assert {ast.literal_eval(key) for key in evidence_dict.keys} == evidence_keys
    writer_source = ast.unparse(_function(tree, "write_evidence"))
    assert "json.dumps(payload, indent=2) + '\\n'" in writer_source


def _validate_selection_and_no_pass(source: str) -> None:
    tree = _probe_tree(source)
    parent = _function(tree, "run_parent")
    parent_source = ast.unparse(parent)
    candidate_loop = next(
        loop
        for loop in ast.walk(parent)
        if isinstance(loop, ast.For)
        and isinstance(loop.iter, ast.Call)
        and ast.unparse(loop.iter.func) == "enumerate"
    )
    assert isinstance(candidate_loop.target, ast.Tuple)
    assert [ast.unparse(element) for element in candidate_loop.target.elts] == [
        "index",
        "candidate",
    ]
    assert [ast.unparse(argument) for argument in candidate_loop.iter.args] == [
        "STANCE_CANDIDATES"
    ]
    assert "subprocess.run(worker_command(args_cli, candidate, worker_output), check=True)" in parent_source
    assert "metrics.append(metrics_from_evidence(evidence))" in parent_source
    assert parent_source.index("metrics.append(metrics_from_evidence(evidence))") < parent_source.index(
        "select_stance_candidate(metrics)"
    )
    assert "except ValueError:" in parent_source
    assert "write_evidence(args_cli.output, metrics, None)" in parent_source
    assert "return 1" in parent_source
    assert "return 0" in parent_source

    worker_source = ast.unparse(_function(tree, "run_worker"))
    assert "candidate = worker_candidate(args_cli)" in worker_source
    assert "metrics = collect_candidate(candidate, args_cli)" in worker_source
    assert "simulation_app.close()" in worker_source
    main_source = ast.unparse(_function(tree, "main"))
    assert "return run_worker(args_cli)" in main_source
    assert "return run_parent(args_cli)" in main_source


def _validate_no_training_dependency(source: str) -> None:
    forbidden = ("checkpoint", "OnPolicyRunner", "rsl_rl", "SmoothActorCritic", "torch.load")
    for literal in forbidden:
        assert literal not in source


def validate_stance_probe_contract(source: str) -> None:
    _validate_runtime_and_candidates(source)
    _validate_environment_lifecycle(source)
    _validate_sampling_and_contacts(source)
    _validate_low_level_step(source)
    _validate_violations_and_evidence(source)
    _validate_selection_and_no_pass(source)
    _validate_no_training_dependency(source)


def test_probe_declares_exact_runtime_and_candidate_contract():
    _validate_runtime_and_candidates(_probe_source())


def test_probe_disables_automatic_terminations_before_each_fresh_environment():
    _validate_environment_lifecycle(_probe_source())


def test_probe_uses_zero_residuals_exact_windows_and_ordered_support_contacts():
    _validate_sampling_and_contacts(_probe_source())


def test_probe_low_level_step_preserves_action_physics_and_sensor_update_order():
    _validate_low_level_step(_probe_source())


def test_probe_records_independent_violations_and_all_selector_evidence():
    _validate_violations_and_evidence(_probe_source())


def test_probe_closes_each_candidate_before_selection_and_exits_nonzero_without_pass():
    _validate_selection_and_no_pass(_probe_source())


def test_probe_has_no_checkpoint_or_policy_dependency():
    _validate_no_training_dependency(_probe_source())


def test_probe_satisfies_composite_source_validator():
    validate_stance_probe_contract(_probe_source())


def test_validator_rejects_extra_control_step_and_manager_computation():
    source = _probe_source()
    mutation = source.replace(
        "        for _ in range(MEASUREMENT_STEPS):\n",
        "        control_step(base_env, zero_actions)\n"
        "        base_env.reward_manager.compute(dt=base_env.step_dt)\n"
        "        base_env.observation_manager.compute()\n\n"
        "        for _ in range(MEASUREMENT_STEPS):\n",
        1,
    )
    assert mutation != source
    with pytest.raises(AssertionError):
        validate_stance_probe_contract(mutation)


@pytest.mark.parametrize(
    "forbidden_call",
    (
        "base_env.reward_manager.compute(dt=base_env.step_dt)",
        "base_env.observation_manager.compute()",
        "base_env.termination_manager.compute()",
        "base_env._reset_idx(torch.arange(NUM_ENVS, device=base_env.device))",
        "base_env.event_manager.apply(mode='interval', dt=base_env.step_dt)",
    ),
)
def test_validator_rejects_manager_reset_and_event_paths_in_low_level_step(forbidden_call: str):
    source = _probe_source()
    anchor = "        base_env.scene.update(dt=base_env.physics_dt)\n"
    mutation = source.replace(anchor, anchor + f"        {forbidden_call}\n", 1)
    assert mutation != source
    with pytest.raises(AssertionError):
        validate_stance_probe_contract(mutation)


def test_validator_rejects_direct_gym_step_inside_sampling_loop():
    source = _probe_source()
    anchor = "            control_step(base_env, zero_actions)\n"
    mutation = source.replace(anchor, anchor + "            gym_env.step(zero_actions.clone())\n", 1)
    assert mutation != source
    with pytest.raises(AssertionError):
        validate_stance_probe_contract(mutation)


def test_validator_rejects_base_environment_step_inside_sampling_loop():
    source = _probe_source()
    anchor = "            control_step(base_env, zero_actions)\n"
    mutation = source.replace(anchor, anchor + "            base_env.step(zero_actions)\n", 1)
    assert mutation != source
    with pytest.raises(AssertionError):
        validate_stance_probe_contract(mutation)


def test_validator_rejects_any_extra_step_inside_low_level_control():
    source = _probe_source()
    anchor = "        base_env.sim.step(render=False)\n"
    mutation = source.replace(anchor, anchor + "        base_env.step(zero_actions)\n", 1)
    assert mutation != source
    with pytest.raises(AssertionError):
        validate_stance_probe_contract(mutation)


@pytest.mark.parametrize(
    ("violation_line", "occurrence"),
    (
        (
            "            body_collision |= bool(torch.any(body_forces > BODY_CONTACT_THRESHOLD_N).item())\n",
            "first",
        ),
        (
            "            body_collision |= bool(torch.any(body_forces > BODY_CONTACT_THRESHOLD_N).item())\n",
            "last",
        ),
        (
            "            low_height |= bool(torch.any(height < LOW_HEIGHT_THRESHOLD_M).item())\n",
            "first",
        ),
        (
            "            low_height |= bool(torch.any(height < LOW_HEIGHT_THRESHOLD_M).item())\n",
            "last",
        ),
    ),
)
def test_validator_requires_each_violation_accumulator_in_both_windows(
    violation_line: str, occurrence: str
):
    source = _probe_source()
    if occurrence == "first":
        mutation = source.replace(violation_line, "", 1)
    else:
        prefix, separator, suffix = source.rpartition(violation_line)
        assert separator
        mutation = prefix + suffix
    assert mutation != source
    with pytest.raises(AssertionError):
        validate_stance_probe_contract(mutation)


@pytest.mark.parametrize(
    "violation_line",
    (
        "            body_collision |= bool(torch.any(body_forces > BODY_CONTACT_THRESHOLD_N).item())\n",
        "            low_height |= bool(torch.any(height < LOW_HEIGHT_THRESHOLD_M).item())\n",
    ),
)
def test_validator_rejects_duplicate_violation_accumulator_in_a_window(violation_line: str):
    source = _probe_source()
    mutation = source.replace(violation_line, violation_line + violation_line, 1)
    assert mutation != source
    with pytest.raises(AssertionError):
        validate_stance_probe_contract(mutation)
