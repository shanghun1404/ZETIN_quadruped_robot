"""Evaluate the flat V10 WASD demo candidate on seven exact commands."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any, Callable, Sequence


TASK_ID = "Isaac-Velocity-Flat-Guide-RobotV2-v10-wasd-evaluation"
PERSISTENT_OUTPUT_ROOT = Path(
    "/home/kang/quadruped_training_smooth_v11/runs/v10_flat_wasd_demo"
)
EVALUATION_SEEDS = (42, 43, 44)
WASD_CASES = (
    ("stop", (0.0, 0.0, 0.0)),
    ("forward", (0.0, 0.5, 0.0)),
    ("reverse", (0.0, -0.5, 0.0)),
    ("yaw_left", (0.0, 0.0, 0.4)),
    ("yaw_right", (0.0, 0.0, -0.4)),
    ("lateral_left", (-0.3, 0.0, 0.0)),
    ("lateral_right", (0.3, 0.0, 0.0)),
)
WARMUP_SECONDS = 1.0
MEASURE_SECONDS = 6.0
ARTIFACT_ACCEPTED = "ACCEPTED"
ARTIFACT_UNACCEPTED = "UNACCEPTED DEMO CANDIDATE"


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _build_parser(*, include_headless: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=_positive_int, default=64)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    if include_headless:
        parser.add_argument("--headless", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser(include_headless=True)
    args = parser.parse_args(argv)
    try:
        args.seeds = tuple(int(value) for value in args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.seeds != EVALUATION_SEEDS:
        parser.error("seeds must be exactly 42 43 44")
    return args


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(key) and _finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return False


def _require_runtime_evidence(report: dict[str, Any]) -> None:
    version = report.get("isaac_version")
    if not isinstance(version, str) or re.fullmatch(r"5\.1\.[0-9]+", version) is None:
        raise ValueError("official Isaac Sim 5.1 evidence is required")
    if not isinstance(report.get("gpu"), str) or not report["gpu"].strip():
        raise ValueError("GPU evidence is required")
    if not isinstance(report.get("job_id"), str) or not report["job_id"].strip():
        raise ValueError("Slurm job evidence is required")


def validate_evaluation_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate immutable evidence and derive all behavioral gates."""
    if not isinstance(report, dict) or not _finite(report):
        raise ValueError("evaluation report must contain only finite values")
    checkpoint = Path(str(report.get("checkpoint", "")))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("evaluation checkpoint must be a regular file")
    actual_sha = sha256_file(checkpoint)
    if report.get("checkpoint_sha256") != actual_sha:
        raise ValueError("checkpoint SHA-256 does not match report")
    if (
        report.get("policy_observation_width") != 48
        or report.get("action_width") != 12
    ):
        raise ValueError("evaluation requires the exact 48/12 interface")
    _require_runtime_evidence(report)
    if report.get("seeds") != list(EVALUATION_SEEDS):
        raise ValueError("evaluation seeds must be exactly 42, 43, and 44")

    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("evaluation case matrix is missing")
    expected_pairs = [
        (seed, name)
        for seed in EVALUATION_SEEDS
        for name, _ in WASD_CASES
    ]
    actual_pairs = [
        (case.get("seed"), case.get("name"))
        for case in cases
        if isinstance(case, dict)
    ]
    if actual_pairs != expected_pairs or len(cases) != len(expected_pairs):
        raise ValueError("evaluation case matrix must contain each seed/command exactly once")

    commands = dict(WASD_CASES)
    failed: list[str] = []
    required_metrics = (
        "survival",
        "body_frame_x_m_s",
        "body_frame_y_m_s",
        "yaw_rate_rad_s",
        "mean_planar_speed_m_s",
        "mean_abs_yaw_rate_rad_s",
        "displacement_m",
    )
    for case in cases:
        seed = int(case["seed"])
        name = str(case["name"])
        if case.get("command") != list(commands[name]):
            raise ValueError(f"command payload mismatch for seed{seed}:{name}")
        if case.get("inference_finite") is not True:
            raise ValueError(f"inference must be finite for seed{seed}:{name}")
        if not all(
            isinstance(case.get(field), (int, float))
            and not isinstance(case.get(field), bool)
            and math.isfinite(float(case[field]))
            for field in required_metrics
        ):
            raise ValueError(f"case metrics must be finite for seed{seed}:{name}")
        if float(case["survival"]) < 0.90:
            failed.append(f"seed{seed}:{name}:survival")
        if name == "forward" and float(case["body_frame_y_m_s"]) < 0.25:
            failed.append(f"seed{seed}:{name}:signed_response")
        elif name == "reverse" and -float(case["body_frame_y_m_s"]) < 0.25:
            failed.append(f"seed{seed}:{name}:signed_response")
        elif name == "yaw_left" and float(case["yaw_rate_rad_s"]) < 0.15:
            failed.append(f"seed{seed}:{name}:signed_response")
        elif name == "yaw_right" and -float(case["yaw_rate_rad_s"]) < 0.15:
            failed.append(f"seed{seed}:{name}:signed_response")
        elif name == "lateral_left" and -float(case["body_frame_x_m_s"]) < 0.12:
            failed.append(f"seed{seed}:{name}:signed_response")
        elif name == "lateral_right" and float(case["body_frame_x_m_s"]) < 0.12:
            failed.append(f"seed{seed}:{name}:signed_response")
        elif name == "stop":
            if float(case["mean_planar_speed_m_s"]) > 0.08:
                failed.append(f"seed{seed}:{name}:planar_speed")
            if float(case["mean_abs_yaw_rate_rad_s"]) > 0.08:
                failed.append(f"seed{seed}:{name}:yaw_rate")
            if float(case["displacement_m"]) > 0.30:
                failed.append(f"seed{seed}:{name}:displacement")

    checked = copy.deepcopy(report)
    checked["failed_gates"] = failed
    checked["passed"] = not failed
    checked["finite"] = True
    checked["shapes_valid"] = True
    checked["artifact_label"] = ARTIFACT_ACCEPTED if not failed else ARTIFACT_UNACCEPTED
    return checked


def write_atomic_report(path: str | Path, report: dict[str, Any]) -> None:
    if not _finite(report):
        raise ValueError("evaluation report must contain only finite values")
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise


def publish_evaluated_artifact(
    checkpoint: str | Path,
    checked_report: dict[str, Any],
) -> Path:
    source = Path(checkpoint)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    if checked_report.get("artifact_label") == ARTIFACT_UNACCEPTED:
        return source
    if checked_report.get("artifact_label") != ARTIFACT_ACCEPTED or checked_report.get("passed") is not True:
        raise ValueError("evaluation artifact label is invalid")
    target = source.with_name("selected.pt")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    temporary = target.with_name(".selected.pt.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return target


def lifecycle_status(run: Callable[[], int], cleanup: Callable[[], Any]) -> int:
    """Preserve rollout status/failure when SimulationApp cleanup exits zero."""
    status = 0
    active_failure = False
    try:
        status = int(run())
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        active_failure = True
        status = 1
    try:
        cleanup()
    except SystemExit as error:
        close_status = 0 if error.code is None else int(error.code)
        if not active_failure and status == 0 and close_status != 0:
            status = close_status
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        if not active_failure and status == 0:
            status = 1
    return status


def _official_isaac_version() -> str:
    from isaacsim.core.version import get_version

    raw = get_version()
    if len(raw) < 5 or (int(raw[2]), int(raw[3])) != (5, 1):
        raise RuntimeError(f"official Isaac Sim 5.1 is required; found {raw!r}")
    return f"{int(raw[2])}.{int(raw[3])}.{int(raw[4])}"


def _observation(env: Any) -> Any:
    value = env.get_observations()
    return value[0] if isinstance(value, tuple) else value


def _rollout_case(
    env: Any,
    policy: Any,
    *,
    seed: int,
    name: str,
    command: Sequence[float],
) -> dict[str, Any]:
    import torch

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    term = raw_env.command_manager.get_term("base_velocity")
    command_tensor = torch.tensor(
        command,
        dtype=torch.float32,
        device=raw_env.device,
    ).repeat(env.num_envs, 1)
    raw_env.reset()
    term.vel_command_b[:] = command_tensor
    obs = _observation(env)
    alive = torch.ones(env.num_envs, device=raw_env.device, dtype=torch.bool)
    warmup_steps = round(WARMUP_SECONDS / raw_env.step_dt)
    measure_steps = round(MEASURE_SECONDS / raw_env.step_dt)

    with torch.no_grad():
        for _ in range(warmup_steps):
            term.vel_command_b[:] = command_tensor
            action = policy(obs)
            if action.shape != (env.num_envs, 12) or not bool(torch.isfinite(action).all()):
                raise RuntimeError(f"non-finite or wrong-shape policy action for {name}")
            step = env.step(action)
            obs, dones = step[0], step[2]
            alive &= ~dones.bool()

        start_xy = robot.data.root_pos_w[:, :2].clone()
        max_displacement = torch.zeros(env.num_envs, device=raw_env.device)
        velocity_samples = []
        yaw_samples = []
        valid_samples = []
        for _ in range(measure_steps):
            term.vel_command_b[:] = command_tensor
            valid_samples.append(alive.clone())
            velocity_samples.append(robot.data.root_lin_vel_b[:, :2].clone())
            yaw_samples.append(robot.data.root_ang_vel_b[:, 2].clone())
            displacement = torch.linalg.vector_norm(
                robot.data.root_pos_w[:, :2] - start_xy,
                dim=1,
            )
            max_displacement = torch.where(
                alive,
                torch.maximum(max_displacement, displacement),
                max_displacement,
            )
            action = policy(obs)
            if action.shape != (env.num_envs, 12) or not bool(torch.isfinite(action).all()):
                raise RuntimeError(f"non-finite or wrong-shape policy action for {name}")
            step = env.step(action)
            obs, dones = step[0], step[2]
            alive &= ~dones.bool()

    valid = torch.stack(valid_samples)
    velocity = torch.stack(velocity_samples)
    yaw = torch.stack(yaw_samples)
    if not bool(valid.any()):
        raise RuntimeError(f"no valid samples for {name}")
    planar_speed = torch.linalg.vector_norm(velocity, dim=-1)
    return {
        "seed": seed,
        "name": name,
        "command": list(command),
        "survival": float(alive.float().mean()),
        "body_frame_x_m_s": float(velocity[..., 0][valid].mean()),
        "body_frame_y_m_s": float(velocity[..., 1][valid].mean()),
        "yaw_rate_rad_s": float(yaw[valid].mean()),
        "mean_planar_speed_m_s": float(planar_speed[valid].mean()),
        "mean_abs_yaw_rate_rad_s": float(yaw[valid].abs().mean()),
        "displacement_m": float(max_displacement.max()),
        "inference_finite": True,
    }


def _evaluate_seeds_in_one_environment(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int, int]:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from v10_wasd_env_cfg import GuideV2V10WASDEnvCfg
    from v10_wasd_ppo_cfg import V10_WASD_PPO_CFG

    env_cfg = GuideV2V10WASDEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = EVALUATION_SEEDS[0]
    env_cfg.episode_length_s = WARMUP_SECONDS + MEASURE_SECONDS + 1.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.curriculum = None
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    if hasattr(env_cfg.events, "base_com"):
        env_cfg.events.base_com = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None
    env_cfg.events.reset_base.params["pose_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "yaw")
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        key: (0.0, 0.0)
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.commands.base_velocity.debug_vis = False
    agent_cfg = copy.deepcopy(V10_WASD_PPO_CFG)
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(args.checkpoint, load_optimizer=False, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        results = []
        for seed in EVALUATION_SEEDS:
            raw_env.reset(seed=seed)
            results.extend(
                _rollout_case(
                    env,
                    policy,
                    seed=seed,
                    name=name,
                    command=command,
                )
                for name, command in WASD_CASES
            )
        observation_width = int(
            raw_env.unwrapped.observation_manager.compute()["policy"].shape[-1]
        )
        action_width = int(raw_env.unwrapped.action_manager.action.shape[-1])
        return results, observation_width, action_width
    finally:
        env.close()


def _validated_runtime_inputs(args: argparse.Namespace) -> tuple[Path, Path, str]:
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output).resolve()
    try:
        output.relative_to(PERSISTENT_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"output must remain beneath {PERSISTENT_OUTPUT_ROOT}") from error
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID is required")
    return checkpoint, output, job_id


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch

    from v10_wasd_env_cfg import GuideV2V10WASDEnvCfg
    from v10_wasd_ppo_cfg import V10_WASD_PPO_CFG

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V10 WASD evaluation")
    checkpoint, output, job_id = _validated_runtime_inputs(args)
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": GuideV2V10WASDEnvCfg,
            "rsl_rl_cfg_entry_point": V10_WASD_PPO_CFG,
        },
    )
    cases, observation_width, action_width = _evaluate_seeds_in_one_environment(args)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_observation_width": observation_width,
        "action_width": action_width,
        "isaac_version": _official_isaac_version(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "job_id": job_id,
        "seeds": list(EVALUATION_SEEDS),
        "cases": cases,
    }
    checked = validate_evaluation_report(report)
    write_atomic_report(output, checked)
    artifact = publish_evaluated_artifact(checkpoint, checked)
    checked["runnable_artifact"] = str(artifact.resolve())
    return checked


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = _build_parser(include_headless=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    try:
        args.seeds = tuple(int(value) for value in args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.seeds != EVALUATION_SEEDS:
        parser.error("seeds must be exactly 42 43 44")
    _validated_runtime_inputs(args)
    simulation_app = AppLauncher(args, fast_shutdown=False).app

    def execute() -> int:
        report = run_evaluation(args)
        return 0 if report["passed"] else 3

    return lifecycle_status(execute, simulation_app.close)


def process_entry(argv: Sequence[str] | None = None) -> None:
    """Flush evaluation evidence and bypass unstable Isaac/Python finalizers."""
    status = 1
    try:
        status = int(main(argv))
    except SystemExit as error:
        status = error.code if isinstance(error.code, int) else 1
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        status = 1
    if status < 0 or status > 255:
        status = 1
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    os._exit(status)


if __name__ == "__main__":
    process_entry()
