"""Portable bundle boundaries: actual files, trusted hashes and relocatable launch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
BUILDER = REPOSITORY / "deploy/jetson_v15/build_bundle.py"


def load_builder():
    assert BUILDER.is_file(), "the portable bundle builder is not implemented"
    spec = importlib.util.spec_from_file_location("v15_bundle_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_file(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def bundle_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)
    for name in ("v14_ros2_control", "v151_ros2_control", "v15_hardware_bridge"):
        write_file(root, f"ros2/{name}/{name}/__init__.py", "")
        write_file(root, f"ros2/{name}/package.xml", f"<package><name>{name}</name></package>\n")
    write_file(root, "ros2/v14_ros2_control/v14_ros2_control/policy.py", "from v13_depth_policy import policy\n")
    write_file(root, "ros2/v15_hardware_bridge/config/hardware.json", '{"enable_motors": false}\n')
    write_file(root, "ros2/v15_hardware_bridge/launch/hardware.launch.py", "")
    write_file(root, "config/v13_depth_policy.py", "from v13_torque_mdp import value\npolicy = value\n")
    write_file(root, "config/v13_torque_mdp.py", "value = 12\n")
    for name in ("v12_vision_student", "v12_training", "v12_camera", "v16_controller_math", "v16_physical_model", "v16_wbc_qp", "v16_force_qp"):
        write_file(root, f"config/{name}.py", "")
    model = write_file(root, "config/experiments/v15_cad_20261011/model/quadruped_cad.xml", "<mujoco/>\n")
    write_file(root, "config/experiments/v15_cad_20261011/model/cad_model.json", json.dumps({"dynamics_mjcf_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "physical_body_count": 13, "source_total_mass_kg": 23.09449815750122}))
    write_file(root, "firmware/stm32g474_usb/README.md", "Board port requires measured calibration.\n")
    write_file(root, "native/v16_wbc/CMakeLists.txt", "# native source fixture\n")
    write_file(root, "native/v16_wbc/wbc_native.cpp", "// native source fixture\n")
    write_file(root, "docs/superpowers/specs/2026-10-12-v15-jetson-usb-design.md", "Approved design\n")
    write_file(root, "docs/superpowers/plans/2026-10-12-v15-jetson-usb.md", "Approved plan\n")
    write_file(root, "reports/v15/cad_rms_20261011/round2/summary.json", '{"scope": "simulation"}\n')
    write_file(root, "reports/v15/cad_rms_20261011/round2/motor_comparison.csv", "joint,rms\nFL_HAA,1\n")
    actor = write_file(root, "actor.pt", "trusted actor bytes\n")
    student = write_file(root, "student.pt", "trusted student bytes\n")
    return root, actor, student


def build(module, source, output):
    root, actor, student = source
    return module.build_bundle(root, output, actor_checkpoint=actor, student_checkpoint=student,
                               actor_sha256=hashlib.sha256(b"trusted actor bytes\n").hexdigest(),
                               student_sha256=hashlib.sha256(b"trusted student bytes\n").hexdigest())


def extract(archive, target):
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(target, filter="data")
    return target / "v15_jetson_20261012"


def test_bundle_copies_transitive_config_dependency_and_checksums(bundle_source, tmp_path):
    module = load_builder()
    archive = tmp_path / "portable.tar.gz"
    build(module, bundle_source, archive)
    bundle = extract(archive, tmp_path / "unpacked")
    assert (bundle / "runtime/config/v13_torque_mdp.py").read_text() == "value = 12\n"
    result = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=bundle, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["checkpoints"]["actor"]["sha256"] == hashlib.sha256(b"trusted actor bytes\n").hexdigest()
    assert manifest["source_repository"]["head"] == subprocess.check_output(["git", "-C", str(bundle_source[0]), "rev-parse", "HEAD"], text=True).strip()


def test_bundle_contains_native_wbc_sources(bundle_source, tmp_path):
    archive=tmp_path/'native.tar.gz'
    build(load_builder(),bundle_source,archive)
    bundle=extract(archive,tmp_path/'native')
    assert (bundle/'native/v16_wbc/wbc_native.cpp').is_file(), 'Native WBC source is missing from the bundle'
    assert (bundle/'native/v16_wbc/CMakeLists.txt').is_file()


def test_unresolved_local_import_aborts_instead_of_shipping_partial_bundle(bundle_source, tmp_path):
    module = load_builder()
    write_file(bundle_source[0], "config/v13_torque_mdp.py", "from v99_missing import value\n")
    archive = tmp_path / "portable.tar.gz"
    with pytest.raises(module.BundleError, match="v99_missing"):
        build(module, bundle_source, archive)
    assert not archive.exists()


def test_training_only_simulator_import_does_not_require_simulator_at_runtime(bundle_source, tmp_path):
    module = load_builder()
    write_file(bundle_source[0], "config/v12_camera.py", "def make_v12_depth_camera_cfg():\n    import isaaclab.sim\n    return isaaclab.sim\n")
    archive = tmp_path / "portable.tar.gz"
    build(module, bundle_source, archive)
    bundle = extract(archive, tmp_path / "unpacked")
    result = subprocess.run(["python3", "-c", "import v12_camera; print(callable(v12_camera.make_v12_depth_camera_cfg))"],
                            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(bundle / "runtime/config")), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"
    write_file(bundle_source[0], "config/v12_camera.py", "import isaaclab.sim\n")
    with pytest.raises(module.BundleError, match="isaaclab"):
        build(module, bundle_source, tmp_path / "broken.tar.gz")


def test_wrong_checkpoint_hash_keeps_previous_archive(bundle_source, tmp_path):
    module = load_builder()
    archive = tmp_path / "portable.tar.gz"
    archive.write_bytes(b"previous verified artifact")
    bundle_source[1].write_text("wrong actor\n")
    with pytest.raises(module.BundleError, match="actor.*SHA256"):
        build(module, bundle_source, archive)
    assert archive.read_bytes() == b"previous verified artifact"


def test_missing_ros_package_is_reported_before_build(bundle_source, tmp_path):
    module = load_builder()
    (bundle_source[0] / "ros2/v15_hardware_bridge/package.xml").unlink()
    with pytest.raises(module.BundleError, match="v15_hardware_bridge"):
        build(module, bundle_source, tmp_path / "portable.tar.gz")


def test_missing_hardware_launch_is_reported_before_build(bundle_source, tmp_path):
    module = load_builder()
    (bundle_source[0] / "ros2/v15_hardware_bridge/launch/hardware.launch.py").unlink()
    with pytest.raises(module.BundleError, match="hardware.launch.py"):
        build(module, bundle_source, tmp_path / "portable.tar.gz")


def test_archive_bytes_ignore_source_mtime(bundle_source, tmp_path):
    module = load_builder()
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    build(module, bundle_source, first)
    for path in bundle_source[0].rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            os.utime(path, (123456789, 123456789))
    build(module, bundle_source, second)
    assert first.read_bytes() == second.read_bytes()


def test_run_script_derives_absolute_paths_after_relocation(bundle_source, tmp_path):
    module = load_builder()
    archive = tmp_path / "portable.tar.gz"
    build(module, bundle_source, archive)
    bundle = extract(archive, tmp_path / "directory with spaces")
    ros_setup = write_file(tmp_path, "jazzy/setup.bash", "true\n")
    write_file(bundle, "workspace/install/setup.bash", "true\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ros2 = write_file(fake_bin, "ros2", "#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    ros2.chmod(0o755)
    environment = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", V15_ROS_SETUP=str(ros_setup))
    result = subprocess.run(["bash", str(bundle / "run.sh")], cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    arguments = json.loads(result.stdout.strip().splitlines()[-1])
    assert arguments[:3] == ["launch", "v15_hardware_bridge", "hardware.launch.py"]
    assert f"actor_checkpoint:={bundle}/checkpoints/model_15699.pt" in arguments
    assert f"student_checkpoint:={bundle}/checkpoints/v12_student_best.pt" in arguments
    assert f"model_path:={bundle}/model/quadruped_cad.xml" in arguments
    assert "enable_motors:=false" in arguments


def test_bundle_verifier_rejects_modified_payload(bundle_source, tmp_path):
    module = load_builder()
    archive = tmp_path / "portable.tar.gz"
    build(module, bundle_source, archive)
    bundle = extract(archive, tmp_path / "unpacked")
    (bundle / "checkpoints/model_15699.pt").write_bytes(b"tampered")
    result = subprocess.run(["python3", str(bundle / "verify_bundle.py")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "model_15699.pt" in result.stderr


def test_environment_checker_records_bsp_and_rejects_wrong_target(tmp_path):
    path = REPOSITORY / "deploy/jetson_v15/check_env.py"
    assert path.is_file(), "target environment checker is not implemented"
    spec = importlib.util.spec_from_file_location("v15_env_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os_release = write_file(tmp_path, "os-release", 'ID=ubuntu\nVERSION_ID="24.04"\n')
    bsp = write_file(tmp_path, "nv_tegra_release", "# R38 fixture BSP\n")
    good = module.inspect_environment(os_release, bsp, machine="aarch64", environ={"ROS_DISTRO": "jazzy"})
    assert good["problems"] == []
    assert good["nvidia_bsp_release"] == "# R38 fixture BSP"
    bad = module.inspect_environment(os_release, bsp, machine="x86_64", environ={"ROS_DISTRO": "humble"})
    assert len(bad["problems"]) == 2


def test_run_rejects_modified_checkpoint_before_ros_launch(bundle_source, tmp_path):
    module = load_builder()
    archive = tmp_path / "portable.tar.gz"
    build(module, bundle_source, archive)
    bundle = extract(archive, tmp_path / "unpacked")
    ros_setup = write_file(tmp_path, "jazzy/setup.bash", "true\n")
    write_file(bundle, "workspace/install/setup.bash", "true\n")
    (bundle / "checkpoints/model_15699.pt").write_text("wrong actor\n")
    result = subprocess.run(["bash", str(bundle / "run.sh")], env=dict(os.environ, V15_ROS_SETUP=str(ros_setup)), capture_output=True, text=True)
    assert result.returncode != 0
    assert "SHA256 mismatch: checkpoints/model_15699.pt" in result.stderr
