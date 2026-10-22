#!/usr/bin/env python3
"""Build the reviewed V15 working-tree snapshot without ROS or simulator imports."""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


BUNDLE_NAME = "v15_jetson_20261012"
ACTOR_SOURCE = "reports/v15/cad_rms_20261011/round2/train_jobs/2341/results/model_15699.pt"
ACTOR_SHA256 = "9a440065d0f93835abbf02278f79fff91aea95a18c3bf4020b83bb092f9e6db6"
STUDENT_SHA256 = "74c0c15644aec802c2637e5054286ea261e0fd6060637506eee61776af3b6f05"
PACKAGES = ("v14_ros2_control", "v151_ros2_control", "v15_hardware_bridge")
CONFIG_SEEDS = ("v13_depth_policy", "v12_vision_student", "v12_training", "v12_camera",
                "v16_controller_math", "v16_physical_model", "v16_wbc_qp", "v16_force_qp")
EXTERNAL_MODULES = {"numpy", "torch", "pinocchio", "serial", "yaml", "rclpy",
                    "sensor_msgs", "nav_msgs", "geometry_msgs", "std_msgs",
                    "builtin_interfaces", "launch", "launch_ros", "ament_index_python",
                    "setuptools", "pytest"}
IGNORE_PARTS = {"__pycache__", ".pytest_cache", ".git", "build", "install", "log"}
EVIDENCE_FILES = (
    "docs/superpowers/specs/2026-10-12-v15-jetson-usb-design.md",
    "docs/superpowers/plans/2026-10-12-v15-jetson-usb.md",
    "reports/v15/cad_rms_20261011/round2/summary.json",
    "reports/v15/cad_rms_20261011/round2/motor_comparison.csv",
)
SUPPORT_FILES = ("README.md", "build_bundle.py", "check_env.py", "verify_bundle.py", "build.sh", "run.sh")
# Existing training factories are retained but never called by the hardware launcher.
DEFERRED_TRAINING_IMPORTS = {
    ("v12_camera", "make_v12_depth_camera_cfg"): "isaaclab",
    ("v13_depth_policy", "make_v13_depth_wrapper"): "isaaclab_rl",
}


class BundleError(RuntimeError):
    """An input is absent, untrusted or cannot form a standalone runtime."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_files(directory):
    if not directory.is_dir():
        raise BundleError(f"required directory is missing: {directory}")
    result = []
    for path in sorted(directory.rglob("*")):
        if set(path.relative_to(directory).parts) & IGNORE_PARTS:
            continue
        if path.is_symlink():
            raise BundleError(f"bundle source must not contain a symlink: {path}")
        if path.is_file():
            result.append(path)
    if not result:
        raise BundleError(f"required directory is empty: {directory}")
    return result


def inspect_imports(path):
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeError) as error:
        raise BundleError(f"cannot inspect Python imports: {path}: {error}") from error
    result = set()
    deferred = set()
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names = {node.module.split(".")[0]}
        else:
            continue
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                allowed = DEFERRED_TRAINING_IMPORTS.get((path.stem, parent.name))
                if allowed in names:
                    names.remove(allowed)
                    deferred.add((parent.name, allowed))
            parent = parents.get(parent)
        result.update(names)
    return result, deferred


def config_closure(repository, package_files):
    """Resolve local imports transitively and reject undeclared dependencies."""
    config = repository / "config"
    pending = set(CONFIG_SEEDS)
    seen = set()
    dependencies = set()
    ignored = EXTERNAL_MODULES | set(sys.stdlib_module_names) | set(PACKAGES)
    for path in package_files:
        if path.suffix == ".py" and not {"test", "tests"} & set(path.parts):
            dependencies.update(inspect_imports(path)[0])
    while pending or dependencies:
        for name in sorted(dependencies):
            if name in ignored:
                continue
            if (config / f"{name}.py").is_file():
                pending.add(name)
            else:
                raise BundleError(f"unresolved runtime import: {name}; add the pure module or declare its external dependency")
        dependencies.clear()
        if not pending:
            break
        name = min(pending)
        pending.remove(name)
        if name in seen:
            continue
        path = config / f"{name}.py"
        if not path.is_file() or path.is_symlink():
            raise BundleError(f"required config module is missing or a symlink: {name}")
        seen.add(name)
        dependencies.update(inspect_imports(path)[0])
    return [config / f"{name}.py" for name in sorted(seen)]


def require_checkpoint(path, expected, role):
    if not path.is_file() or path.is_symlink():
        raise BundleError(f"{role} checkpoint is missing or a symlink: {path}")
    actual = sha256(path)
    if actual != expected:
        raise BundleError(f"{role} checkpoint SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def build_bundle(repository, output, *, student_checkpoint, actor_checkpoint=None,
                 actor_sha256=ACTOR_SHA256, student_sha256=STUDENT_SHA256):
    repository = Path(repository).resolve()
    output = Path(output).resolve()
    actor = Path(actor_checkpoint).resolve() if actor_checkpoint else repository / ACTOR_SOURCE
    student = Path(student_checkpoint).resolve()
    actor_hash = require_checkpoint(actor, actor_sha256, "actor")
    student_hash = require_checkpoint(student, student_sha256, "student")
    for relative in ("ros2/v15_hardware_bridge/launch/hardware.launch.py",
                     "ros2/v15_hardware_bridge/config/hardware.json"):
        if not (repository / relative).is_file():
            raise BundleError(f"required runtime asset is missing: {relative}")
    sources = {}
    package_files = []
    for name in PACKAGES:
        directory = repository / "ros2" / name
        if not (directory / "package.xml").is_file():
            raise BundleError(f"required ROS package is missing: {name}")
        for path in regular_files(directory):
            sources[f"workspace/src/{name}/{path.relative_to(directory).as_posix()}"] = (path, path.relative_to(repository).as_posix())
            package_files.append(path)
    closure = config_closure(repository, package_files)
    for path in closure:
        sources[f"runtime/config/{path.name}"] = (path, path.relative_to(repository).as_posix())
    model_directory = repository / "config/experiments/v15_cad_20261011/model"
    for path in regular_files(model_directory):
        sources[f"model/{path.relative_to(model_directory).as_posix()}"] = (path, path.relative_to(repository).as_posix())
    metadata = json.loads((model_directory / "cad_model.json").read_text())
    if sha256(model_directory / "quadruped_cad.xml") != metadata["dynamics_mjcf_sha256"]:
        raise BundleError("CAD model SHA256 does not match cad_model.json")
    firmware = repository / "firmware/stm32g474_usb"
    for path in regular_files(firmware):
        sources[f"firmware/stm32g474_usb/{path.relative_to(firmware).as_posix()}"] = (path, path.relative_to(repository).as_posix())
    for relative in EVIDENCE_FILES:
        path = repository / relative
        if not path.is_file():
            raise BundleError(f"required source evidence is missing: {relative}")
        sources[f"evidence/{relative}"] = (path, relative)
    for name in SUPPORT_FILES:
        path = Path(__file__).resolve().parent / name
        if not path.is_file():
            raise BundleError(f"bundle support file is missing: {name}")
        sources[name] = (path, f"deploy/jetson_v15/{name}")
    sources["checkpoints/model_15699.pt"] = (actor, "trusted V15 actor model_15699.pt")
    sources["checkpoints/v12_student_best.pt"] = (student, "trusted frozen V12 DAgger best.pt")
    try:
        head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise BundleError("source repository HEAD is unavailable") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v15-bundle-", dir=output.parent) as temporary:
        staging = Path(temporary) / BUNDLE_NAME
        staging.mkdir()
        entries = []
        for relative, (source, identity) in sorted(sources.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            entries.append({"path": relative, "source": identity, "sha256": sha256(destination), "size_bytes": destination.stat().st_size})
        manifest = {
            "schema": 1, "bundle": BUNDLE_NAME,
            "source_repository": {"head": head, "snapshot": "working tree files identified by the per-file SHA256 values"},
            "target": {"architecture": "aarch64", "ubuntu": "24.04", "ros": "jazzy", "board": "NUCLEO-G474RE (MB1367)"},
            "default_enable_motors": False,
            "checkpoints": {
                "actor": {"path": "checkpoints/model_15699.pt", "sha256": actor_hash, "source": ACTOR_SOURCE},
                "student": {"path": "checkpoints/v12_student_best.pt", "sha256": student_hash, "source": "frozen V12 DAgger best.pt"},
            },
            "physical_model": {"path": "model/quadruped_cad.xml", "sha256": metadata["dynamics_mjcf_sha256"],
                               "physical_body_count": metadata["physical_body_count"], "source_total_mass_kg": metadata["source_total_mass_kg"],
                               "metadata": "model/cad_model.json"},
            "config_modules": [path.stem for path in closure],
            "deferred_training_imports": [
                {"source": path.relative_to(repository).as_posix(), "function": function, "module": module}
                for path in closure for function, module in sorted(inspect_imports(path)[1])
            ],
            "simulation_report": "evidence/reports/v15/cad_rms_20261011/round2/summary.json",
            "validation_scope": "software preparation; Jetson timing, real MCU USB, motor CAN and sensor drivers require hardware validation",
            "files": entries,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(staging).as_posix()}\n" for path in files))
        temporary_archive = Path(temporary) / "bundle.tar.gz"
        with temporary_archive.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for path in [staging, *sorted(staging.rglob("*"))]:
                        info = archive.gettarinfo(str(path), arcname=path.relative_to(staging.parent).as_posix())
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ""
                        info.mode = 0o755 if path.is_dir() or path.suffix == ".sh" else 0o644
                        info.pax_headers = {}
                        if path.is_file():
                            with path.open("rb") as stream:
                                archive.addfile(info, stream)
                        else:
                            archive.addfile(info)
        os.replace(temporary_archive, output)
    return {"archive": str(output), "sha256": sha256(output), "file_count": len(entries) + 2, "config_modules": manifest["config_modules"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--student-checkpoint", type=Path, required=True, help="frozen V12 best.pt; trusted SHA256 is enforced")
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.repository / "reports/v15/jetson_usb_20261012/v15_jetson_20261012.tar.gz"
    try:
        result = build_bundle(args.repository, output, student_checkpoint=args.student_checkpoint, actor_checkpoint=args.actor_checkpoint)
    except (BundleError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"bundle build failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
