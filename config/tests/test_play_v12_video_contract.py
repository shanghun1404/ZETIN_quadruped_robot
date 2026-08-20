"""Source-boundary contract for the V12 closed-loop video recorder."""

from __future__ import annotations

import ast
from pathlib import Path


PLAYER = Path("play_v12_stairs_video.py")


def _source() -> str:
    return PLAYER.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _main() -> ast.FunctionDef:
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def test_v12_video_runs_the_depth_student_and_frozen_actor_closed_loop():
    source = _source()

    assert "make_v12_depth_camera_cfg(step_dt)" in source
    assert "depth = camera_depth_nchw(depth_camera)" in source
    assert "predicted_scan = student(depth)" in source
    assert "student_observation = reconstruct_teacher_observation(" in source
    assert "student_action = frozen_actor(student_observation)" in source
    assert "env.step(student_action)" in source
    assert "env.step(teacher_action)" not in source
    assert "teacher_policy(observations)" in source
    assert "actor_equivalence_error > 1.0e-6" in source


def test_v12_video_records_one_deterministic_stair_rollout_with_tracking_view():
    source = _source()

    assert 'choices=("stairs_up", "stairs_down")' in source
    assert 'choices=("side", "rear_quarter")' in source
    assert "env_cfg.scene.num_envs = 1" in source
    assert "env_cfg.observations.policy.enable_corruption = False" in source
    assert "env_cfg.curriculum = None" in source
    assert "env_cfg.events.push_robot = None" in source
    assert "pin_terrain(raw_env, args_cli.stair_level)" in source
    assert "class TrackingWrapper(gym.Wrapper):" in source
    assert "gym.wrappers.RecordVideo(" in source
    assert "step_trigger=lambda step: step == 0" in source
    assert "create_fresh_recording_dir(output_dir, prefix)" in source
    assert "copy_exact_recording(record_dir, prefix, final_path)" in source
    assert 'f"v12_{student_name}_{args_cli.scenario}' in source


def test_v12_video_writes_hash_bound_telemetry_and_closes_resources():
    source = _source()
    main = _main()

    assert '"teacher_checkpoint_sha256": sha256_file(args_cli.teacher_checkpoint)' in source
    assert '"student_checkpoint_sha256": sha256_file(args_cli.student_checkpoint)' in source
    assert '"policy_path": "depth_to_vit_to_frozen_v11_actor"' in source
    assert '"video": str(final_path)' in source
    assert '"video_sha256": sha256_file(final_path)' in source
    assert not any(isinstance(node, ast.ExceptHandler) for node in ast.walk(main))
    cleanup = next(node for node in ast.walk(main) if isinstance(node, ast.Try))
    assert [ast.unparse(statement) for statement in cleanup.finalbody] == ["env.close()"]


def test_v12_video_enables_both_rgb_recording_and_depth_sensor():
    source = _source()

    assert "args_cli.enable_cameras = True" in source
    assert "args_cli.headless = True" in source
    assert 'render_mode="rgb_array"' in source
    assert 'depth_camera = env.unwrapped.scene["depth_camera"]' in source
    assert "args_cli.video_length <= 0" in source
    assert "args_cli.speed <= 0.0" in source
