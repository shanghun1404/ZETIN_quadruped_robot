"""Pure contracts for V10 WASD training, evaluation, and artifact labels."""

from __future__ import annotations

import hashlib
import json
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import train_v10_wasd as trainer
from evaluate_v10_wasd import (
    ARTIFACT_ACCEPTED,
    ARTIFACT_UNACCEPTED,
    EVALUATION_SEEDS,
    WASD_CASES,
    lifecycle_status,
    publish_evaluated_artifact,
    validate_evaluation_report,
    write_atomic_report,
)


def _case(seed: int, name: str, command: tuple[float, float, float]) -> dict[str, object]:
    values = {
        "stop": (0.0, 0.0, 0.0),
        "forward": (0.0, 0.35, 0.0),
        "reverse": (0.0, -0.35, 0.0),
        "yaw_left": (0.0, 0.0, 0.22),
        "yaw_right": (0.0, 0.0, -0.22),
        "lateral_left": (-0.18, 0.0, 0.0),
        "lateral_right": (0.18, 0.0, 0.0),
    }
    body_x, body_y, yaw = values[name]
    return {
        "seed": seed,
        "name": name,
        "command": list(command),
        "survival": 0.95,
        "body_frame_x_m_s": body_x,
        "body_frame_y_m_s": body_y,
        "yaw_rate_rad_s": yaw,
        "mean_planar_speed_m_s": 0.03 if name == "stop" else 0.35,
        "mean_abs_yaw_rate_rad_s": 0.02 if name == "stop" else abs(yaw),
        "displacement_m": 0.12 if name == "stop" else 2.0,
        "inference_finite": True,
    }


def _valid_report(checkpoint: Path) -> dict[str, object]:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "policy_observation_width": 48,
        "action_width": 12,
        "isaac_version": "5.1.3",
        "gpu": "NVIDIA GeForce RTX 5070",
        "job_id": "901",
        "seeds": list(EVALUATION_SEEDS),
        "cases": [
            _case(seed, name, command)
            for seed in EVALUATION_SEEDS
            for name, command in WASD_CASES
        ],
    }


def test_trainer_allows_only_the_pinned_v10_4_checkpoint_hash(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_2000.pt"
    checkpoint.write_bytes(b"v10 checkpoint fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="SHA-256"):
        trainer.verify_checkpoint_hash(checkpoint)

    monkeypatch.setattr(trainer, "EXPECTED_V10_4_SHA256", digest)
    assert trainer.verify_checkpoint_hash(checkpoint) == digest
    assert trainer.EXPECTED_V10_4_SHA256 == digest


def test_trainer_refuses_output_root_escape_root_itself_and_existing_child(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(trainer, "PERSISTENT_OUTPUT_ROOT", root)

    with pytest.raises(ValueError, match="new child"):
        trainer.validate_new_output_directory(root)
    with pytest.raises(ValueError, match="beneath"):
        trainer.validate_new_output_directory(tmp_path / "outside")
    existing = root / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        trainer.validate_new_output_directory(existing)
    assert trainer.validate_new_output_directory(root / "new") == (root / "new").resolve()


def test_final_checkpoint_selection_requires_the_exact_expected_iteration(tmp_path):
    (tmp_path / "model_2001.pt").write_bytes(b"one")
    (tmp_path / "model_2002.pt").write_bytes(b"two")
    (tmp_path / "model_latest.pt").write_bytes(b"ignored")

    assert trainer.select_final_checkpoint(tmp_path, expected_iteration=2002).name == "model_2002.pt"
    with pytest.raises(RuntimeError, match="iteration mismatch"):
        trainer.select_final_checkpoint(tmp_path, expected_iteration=2003)


def test_training_publishes_only_a_finite_shape_compatible_demo_candidate(tmp_path):
    final = tmp_path / "model_2002.pt"
    final.write_bytes(b"finite checkpoint")
    report = {"finite": True, "shapes_valid": True}

    candidate = trainer.publish_training_candidate(final, tmp_path, report)

    assert candidate.name == "demo_candidate.pt"
    assert candidate.read_bytes() == final.read_bytes()
    assert not (tmp_path / "selected.pt").exists()
    with pytest.raises(FileExistsError):
        trainer.publish_training_candidate(final, tmp_path, report)

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RuntimeError, match="finite and shape-compatible"):
        trainer.publish_training_candidate(
            final,
            other,
            {"finite": False, "shapes_valid": True},
        )


def test_evaluation_accepts_only_the_exact_three_seed_seven_command_matrix(tmp_path):
    checkpoint = tmp_path / "demo_candidate.pt"
    checkpoint.write_bytes(b"candidate")

    checked = validate_evaluation_report(_valid_report(checkpoint))

    assert checked["passed"] is True
    assert checked["failed_gates"] == []
    assert checked["artifact_label"] == ARTIFACT_ACCEPTED
    assert len(checked["cases"]) == 21

    missing = _valid_report(checkpoint)
    missing["cases"] = missing["cases"][:-1]
    with pytest.raises(ValueError, match="matrix"):
        validate_evaluation_report(missing)

    duplicate = _valid_report(checkpoint)
    duplicate["cases"][-1] = dict(duplicate["cases"][0])
    with pytest.raises(ValueError, match="matrix"):
        validate_evaluation_report(duplicate)


@pytest.mark.parametrize(
    ("name", "metric", "bad_value", "gate_suffix"),
    [
        ("forward", "body_frame_y_m_s", 0.24, "signed_response"),
        ("reverse", "body_frame_y_m_s", -0.24, "signed_response"),
        ("yaw_left", "yaw_rate_rad_s", 0.14, "signed_response"),
        ("yaw_right", "yaw_rate_rad_s", -0.14, "signed_response"),
        ("lateral_left", "body_frame_x_m_s", -0.11, "signed_response"),
        ("lateral_right", "body_frame_x_m_s", 0.11, "signed_response"),
        ("stop", "mean_planar_speed_m_s", 0.081, "planar_speed"),
        ("stop", "mean_abs_yaw_rate_rad_s", 0.081, "yaw_rate"),
        ("stop", "displacement_m", 0.301, "displacement"),
    ],
)
def test_behavioral_gate_failure_is_visible_but_remains_a_runnable_candidate(
    tmp_path, name, metric, bad_value, gate_suffix
):
    checkpoint = tmp_path / "demo_candidate.pt"
    checkpoint.write_bytes(b"candidate")
    report = _valid_report(checkpoint)
    case = next(
        item for item in report["cases"]
        if item["seed"] == 42 and item["name"] == name
    )
    case[metric] = bad_value

    checked = validate_evaluation_report(report)

    assert checked["passed"] is False
    assert checked["artifact_label"] == ARTIFACT_UNACCEPTED
    assert checked["failed_gates"] == [f"seed42:{name}:{gate_suffix}"]


def test_shape_finiteness_and_survival_fail_closed(tmp_path):
    checkpoint = tmp_path / "demo_candidate.pt"
    checkpoint.write_bytes(b"candidate")

    wrong_shape = _valid_report(checkpoint)
    wrong_shape["policy_observation_width"] = 47
    with pytest.raises(ValueError, match="48/12"):
        validate_evaluation_report(wrong_shape)

    non_finite = _valid_report(checkpoint)
    non_finite["cases"][0]["body_frame_x_m_s"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_evaluation_report(non_finite)

    low_survival = _valid_report(checkpoint)
    low_survival["cases"][0]["survival"] = 0.89
    checked = validate_evaluation_report(low_survival)
    assert checked["failed_gates"] == ["seed42:stop:survival"]


def test_report_checkpoint_identity_and_runtime_evidence_are_mandatory(tmp_path):
    checkpoint = tmp_path / "demo_candidate.pt"
    checkpoint.write_bytes(b"candidate")

    for field, bad_value, message in (
        ("checkpoint_sha256", "0" * 64, "checkpoint SHA-256"),
        ("isaac_version", "5.0.0", "Isaac Sim 5.1"),
        ("gpu", "", "GPU"),
        ("job_id", "", "job"),
        ("seeds", [42, 43], "seeds"),
    ):
        report = _valid_report(checkpoint)
        report[field] = bad_value
        with pytest.raises(ValueError, match=message):
            validate_evaluation_report(report)


def test_atomic_report_refuses_existing_or_symlink_targets(tmp_path):
    checkpoint = tmp_path / "demo_candidate.pt"
    checkpoint.write_bytes(b"candidate")
    report = validate_evaluation_report(_valid_report(checkpoint))
    target = tmp_path / "evaluation.json"

    write_atomic_report(target, report)

    assert json.loads(target.read_text())["artifact_label"] == ARTIFACT_ACCEPTED
    with pytest.raises(FileExistsError):
        write_atomic_report(target, report)

    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        write_atomic_report(link, report)


def test_evaluated_artifact_selects_only_a_pass_and_preserves_failed_candidate(tmp_path):
    checkpoint = tmp_path / "demo_candidate.pt"
    checkpoint.write_bytes(b"candidate")
    passed = validate_evaluation_report(_valid_report(checkpoint))

    selected = publish_evaluated_artifact(checkpoint, passed)

    assert selected.name == "selected.pt"
    assert selected.read_bytes() == checkpoint.read_bytes()

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    failed_checkpoint = failed_dir / "demo_candidate.pt"
    failed_checkpoint.write_bytes(b"failed candidate")
    failed_report = _valid_report(failed_checkpoint)
    failed_report["cases"][0]["survival"] = 0.1
    checked = validate_evaluation_report(failed_report)
    assert publish_evaluated_artifact(failed_checkpoint, checked) == failed_checkpoint
    assert not (failed_dir / "selected.pt").exists()


def test_active_rollout_failure_cannot_be_replaced_by_cleanup_exit_zero():
    def rollout():
        raise RuntimeError("rollout exploded")

    def cleanup():
        raise SystemExit(0)

    assert lifecycle_status(rollout, cleanup) == 1
    assert lifecycle_status(lambda: 3, cleanup) == 3
    assert trainer.lifecycle_status(rollout, cleanup) == 1


def test_training_cli_requires_positive_counts_and_exact_paths():
    args = trainer.parse_args(
        [
            "--num_envs", "256",
            "--seed", "42",
            "--max_iterations", "2",
            "--checkpoint", "/checkpoints/model_2000.pt",
            "--output-root", "/runs/smoke",
            "--headless",
        ]
    )
    assert args.num_envs == 256
    assert args.seed == 42
    assert args.max_iterations == 2
    assert args.headless is True
    with pytest.raises(SystemExit):
        trainer.parse_args(
            [
                "--num_envs", "0",
                "--seed", "42",
                "--max_iterations", "2",
                "--checkpoint", "/checkpoints/model_2000.pt",
                "--output-root", "/runs/smoke",
            ]
        )


def test_isaac_entrypoints_disable_fast_shutdown_at_the_launcher_boundary(monkeypatch):
    calls: list[dict[str, object]] = []
    app_module = ModuleType("isaaclab.app")

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")

        def __init__(self, args, **kwargs):
            calls.append(kwargs)
            self.app = SimpleNamespace(close=lambda: None)

    app_module.AppLauncher = AppLauncher
    isaaclab_module = ModuleType("isaaclab")
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setattr(trainer, "verify_checkpoint_hash", lambda path: "a" * 64)
    monkeypatch.setattr(trainer, "validate_new_output_directory", lambda path: Path(path))
    monkeypatch.setattr(trainer, "run_training", lambda args: {"finite": True})

    import evaluate_v10_wasd as evaluator

    monkeypatch.setattr(
        evaluator,
        "_validated_runtime_inputs",
        lambda args: (Path(args.checkpoint), Path(args.output), "901"),
    )
    monkeypatch.setattr(
        evaluator,
        "run_evaluation",
        lambda args: {"passed": False},
    )

    assert trainer.main(
        [
            "--num_envs", "1", "--seed", "42", "--max_iterations", "1",
            "--checkpoint", "/checkpoint.pt", "--output-root", "/run", "--headless",
        ]
    ) == 0
    assert evaluator.main(
        [
            "--num_envs", "1", "--seeds", "42", "43", "44",
            "--checkpoint", "/candidate.pt", "--output", "/evaluation.json", "--headless",
        ]
    ) == 3
    assert calls == [
        {"fast_shutdown": False},
        {"fast_shutdown": False},
    ]


def test_training_process_entry_hard_exits_after_cleanup_without_python_finalizers(monkeypatch):
    exits: list[int] = []

    class ExitCalled(Exception):
        pass

    def hard_exit(status):
        exits.append(status)
        raise ExitCalled

    monkeypatch.setattr(trainer.os, "_exit", hard_exit)
    monkeypatch.setattr(trainer, "main", lambda argv=None: 0)
    with pytest.raises(ExitCalled):
        trainer.process_entry([])
    assert exits == [0]

    exits.clear()
    monkeypatch.setattr(
        trainer,
        "main",
        lambda argv=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(ExitCalled):
        trainer.process_entry([])
    assert exits == [1]


def test_evaluation_process_entry_hard_exits_after_cleanup_without_python_finalizers(monkeypatch):
    import evaluate_v10_wasd as evaluator

    exits: list[int] = []

    class ExitCalled(Exception):
        pass

    def hard_exit(status):
        exits.append(status)
        raise ExitCalled

    monkeypatch.setattr(evaluator.os, "_exit", hard_exit)
    monkeypatch.setattr(evaluator, "main", lambda argv=None: 3)
    with pytest.raises(ExitCalled):
        evaluator.process_entry([])
    assert exits == [3]

    exits.clear()
    monkeypatch.setattr(
        evaluator,
        "main",
        lambda argv=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(ExitCalled):
        evaluator.process_entry([])
    assert exits == [1]


def test_rollout_preserves_the_rsl_observation_group_mapping():
    import evaluate_v10_wasd as evaluator

    source = inspect.getsource(evaluator._rollout_case)
    assert "obs = _observation(env)" in source
    assert 'observation_manager.compute()["policy"]' not in source


def test_all_evaluation_seeds_reuse_one_isaac_environment():
    import evaluate_v10_wasd as evaluator

    run_source = inspect.getsource(evaluator.run_evaluation)
    worker_source = inspect.getsource(evaluator._evaluate_seeds_in_one_environment)
    assert "_evaluate_seeds_in_one_environment(args)" in run_source
    assert "for seed in EVALUATION_SEEDS" in worker_source
    assert "raw_env.reset(seed=seed)" in worker_source
    assert "_evaluate_seed(args, seed)" not in run_source
