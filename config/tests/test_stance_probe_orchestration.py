import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from stance_calibration import STANCE_CANDIDATES, TARGET_HEIGHT_M, StanceCandidate


PROBE_PATH = Path("probe_stance_calibration.py")


def _source() -> str:
    return PROBE_PATH.read_text()


def _tree() -> ast.Module:
    return ast.parse(_source())


def _function(name: str) -> ast.FunctionDef:
    return next(node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == name)


def _load_parent_safe_probe():
    tree = _tree()
    top_level_isaac_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaac"))
            or (
                isinstance(node, ast.Import)
                and any(alias.name.startswith("isaac") for alias in node.names)
            )
        )
    ]
    assert top_level_isaac_imports == []
    spec = importlib.util.spec_from_file_location("probe_stance_calibration_for_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evidence(candidate: StanceCandidate, passed: bool) -> dict[str, object]:
    return {
        "half_angle_rad": candidate.half_angle_rad,
        "hfe_rad": candidate.hfe_rad,
        "kfe_rad": candidate.kfe_rad,
        "spawn_height_m": candidate.spawn_height_m,
        "mean_height_m": TARGET_HEIGHT_M if passed else 0.40,
        "support_fraction": 1.0 if passed else 0.0,
        "rms_vertical_velocity_m_s": 0.01 if passed else 0.5,
        "p95_height_range_m": 0.01 if passed else 0.2,
        "body_collision": False,
        "low_height": False,
        "passed": passed,
    }


def _args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(num_envs=64, output=str(output), headless=True, device="cuda:1")


def test_parent_ast_owns_no_app_or_environment_and_worker_owns_exactly_one():
    parent = _function("run_parent")
    parent_source = ast.unparse(parent)
    assert "AppLauncher" not in parent_source
    assert "gym.make" not in parent_source
    assert "TemporaryDirectory" in parent_source
    assert "subprocess.run" in parent_source
    assert "check=True" in parent_source

    worker = _function("run_worker")
    worker_source = ast.unparse(worker)
    assert worker_source.count("AppLauncher(args_cli)") == 1
    assert worker_source.count("collect_candidate(candidate, args_cli)") == 1
    assert "simulation_app.close()" in worker_source
    assert len(
        [
            call
            for call in ast.walk(_function("collect_candidate"))
            if isinstance(call, ast.Call) and ast.unparse(call.func) == "gym.make"
        ]
    ) == 1


def test_parent_runs_exact_workers_in_order_forwards_runtime_args_and_cleans_tempdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe = _load_parent_safe_probe()
    commands: list[list[str]] = []
    worker_outputs: list[Path] = []

    def fake_run(command: list[str], check: bool):
        assert check is True
        commands.append(command)
        half_angle = float(command[command.index("--worker_half_angle_rad") + 1])
        candidate = next(item for item in STANCE_CANDIDATES if item.half_angle_rad == half_angle)
        output = Path(command[command.index("--worker_output") + 1])
        worker_outputs.append(output)
        output.write_text(json.dumps(_evidence(candidate, passed=half_angle == 0.87)) + "\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    output = tmp_path / "final.json"
    assert probe.run_parent(_args(output)) == 0

    assert [
        float(command[command.index("--worker_half_angle_rad") + 1]) for command in commands
    ] == [candidate.half_angle_rad for candidate in STANCE_CANDIDATES]
    assert len({str(path) for path in worker_outputs}) == 4
    for command, worker_output in zip(commands, worker_outputs):
        assert command[:2] == [sys.executable, str(PROBE_PATH.resolve())]
        assert command[command.index("--worker_output") + 1] == str(worker_output)
        assert command[command.index("--num_envs") + 1] == "64"
        assert command[command.index("--device") + 1] == "cuda:1"
        assert "--headless" in command
        index = commands.index(command)
        candidate = STANCE_CANDIDATES[index]
        assert float(command[command.index("--worker_hfe_rad") + 1]) == candidate.hfe_rad
        assert float(command[command.index("--worker_kfe_rad") + 1]) == candidate.kfe_rad
        assert (
            float(command[command.index("--worker_spawn_height_m") + 1])
            == candidate.spawn_height_m
        )
    assert not worker_outputs[0].parent.exists()

    payload = json.loads(output.read_text())
    assert payload["target_height_m"] == 0.35
    assert [item["half_angle_rad"] for item in payload["candidates"]] == [
        candidate.half_angle_rad for candidate in STANCE_CANDIDATES
    ]
    assert payload["selected_candidate"]["half_angle_rad"] == 0.87


def test_parent_writes_all_ordered_evidence_and_returns_nonzero_when_none_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe = _load_parent_safe_probe()

    def fake_run(command: list[str], check: bool):
        assert check is True
        half_angle = float(command[command.index("--worker_half_angle_rad") + 1])
        candidate = next(item for item in STANCE_CANDIDATES if item.half_angle_rad == half_angle)
        output = Path(command[command.index("--worker_output") + 1])
        output.write_text(json.dumps(_evidence(candidate, passed=False)) + "\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    output = tmp_path / "no-pass.json"
    assert probe.run_parent(_args(output)) == 1
    payload = json.loads(output.read_text())
    assert [item["half_angle_rad"] for item in payload["candidates"]] == [
        candidate.half_angle_rad for candidate in STANCE_CANDIDATES
    ]
    assert payload["selected_candidate"] is None


def test_parent_propagates_worker_failure_and_still_cleans_tempdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    probe = _load_parent_safe_probe()
    worker_outputs: list[Path] = []

    def fake_run(command: list[str], check: bool):
        assert check is True
        worker_outputs.append(Path(command[command.index("--worker_output") + 1]))
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        probe.run_parent(_args(tmp_path / "never-written.json"))
    assert len(worker_outputs) == 1
    assert not worker_outputs[0].parent.exists()
