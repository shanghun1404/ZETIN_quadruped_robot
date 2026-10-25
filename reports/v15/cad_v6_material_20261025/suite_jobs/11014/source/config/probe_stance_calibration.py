from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from stance_calibration import (
    STANCE_CANDIDATES,
    TARGET_HEIGHT_M,
    StanceCandidate,
    StanceMetrics,
    passes_stance,
    select_stance_candidate,
)


NUM_ENVS = 64
SETTLE_STEPS = 50
MEASUREMENT_STEPS = 50
CONTROL_STEP_S = 0.02
FOOT_CONTACT_THRESHOLD_N = 5.0
BODY_CONTACT_THRESHOLD_N = 1.0
LOW_HEIGHT_THRESHOLD_M = 0.15
TASK_ID = "Isaac-Velocity-Flat-Guide-Stance-Calibration-v4"


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Calibrate the v4 zero-residual standing height.")
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument("--output", default="stance_calibration_v4.json")
    parser.add_argument("--worker_half_angle_rad", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_hfe_rad", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_kfe_rad", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_spawn_height_m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_output", default=None, help=argparse.SUPPRESS)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def control_step(base_env, zero_actions) -> None:
    base_env.action_manager.process_action(zero_actions.to(base_env.device))
    for _ in range(base_env.cfg.decimation):
        base_env._sim_step_counter += 1
        base_env.action_manager.apply_action()
        base_env.scene.write_data_to_sim()
        base_env.sim.step(render=False)
        base_env.scene.update(dt=base_env.physics_dt)
    base_env.episode_length_buf += 1
    base_env.common_step_counter += 1


def collect_candidate(candidate: StanceCandidate, args_cli: argparse.Namespace) -> StanceMetrics:
    import gymnasium as gym
    import torch
    from isaaclab.managers import SceneEntityCfg

    from smooth_env_cfg import GroundedGuideEnvCfg
    import smooth_mdp

    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": GroundedGuideEnvCfg},
        )

    env_cfg = GroundedGuideEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = 42
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, candidate.spawn_height_m)
    env_cfg.scene.robot.init_state.joint_pos[".*_HFE"] = candidate.hfe_rad
    env_cfg.scene.robot.init_state.joint_pos[".*_KFE"] = candidate.kfe_rad
    env_cfg.terminations.time_out = None
    env_cfg.terminations.base_contact = None
    env_cfg.terminations.low_height = None
    env_cfg.terminations.support_loss = None
    env_cfg.curriculum.velocity_commands = None
    env_cfg.curriculum.grounded_trot_contact = None
    deterministic_base_ranges = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.events.reset_base.params["pose_range"] = deterministic_base_ranges
    env_cfg.events.reset_base.params["velocity_range"] = deterministic_base_ranges
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.physics_material.params["static_friction_range"] = (0.9, 0.9)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.7, 0.7)
    gym_env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        gym_env.reset()
        base_env = gym_env.unwrapped
        if not math.isclose(base_env.step_dt, CONTROL_STEP_S, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"control step must be {CONTROL_STEP_S} s, got {base_env.step_dt}")

        ordered_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_sensor = SceneEntityCfg(
            "contact_forces", body_names=ordered_feet, preserve_order=True
        )
        body_sensor = SceneEntityCfg("contact_forces", body_names="base_link|.*_hip")
        ordered_sensor.resolve(base_env.scene)
        body_sensor.resolve(base_env.scene)
        zero_actions = torch.zeros(
            (NUM_ENVS, base_env.action_manager.total_action_dim), device=base_env.device
        )
        body_collision = False
        low_height = False

        for _ in range(SETTLE_STEPS):
            control_step(base_env, zero_actions)
            height = smooth_mdp.local_base_height_observation(base_env).squeeze(-1)
            body_forces = smooth_mdp._contact_forces(base_env, body_sensor)
            body_collision |= bool(torch.any(body_forces > BODY_CONTACT_THRESHOLD_N).item())
            low_height |= bool(torch.any(height < LOW_HEIGHT_THRESHOLD_M).item())

        robot = base_env.scene["robot"]
        height_min = torch.full((NUM_ENVS,), torch.inf, device=base_env.device)
        height_max = torch.full((NUM_ENVS,), -torch.inf, device=base_env.device)
        sum_height = 0.0
        sum_vertical_velocity_sq = 0.0
        support_samples = 0
        total_samples = NUM_ENVS * MEASUREMENT_STEPS

        for _ in range(MEASUREMENT_STEPS):
            control_step(base_env, zero_actions)
            height = smooth_mdp.local_base_height_observation(base_env).squeeze(-1)
            body_forces = smooth_mdp._contact_forces(base_env, body_sensor)
            contacts = smooth_mdp.foot_contact_flags(
                base_env, FOOT_CONTACT_THRESHOLD_N, ordered_sensor
            )
            contact_count = contacts.sum(dim=-1)
            body_collision |= bool(torch.any(body_forces > BODY_CONTACT_THRESHOLD_N).item())
            low_height |= bool(torch.any(height < LOW_HEIGHT_THRESHOLD_M).item())
            height_min = torch.minimum(height_min, height)
            height_max = torch.maximum(height_max, height)
            sum_height += height.sum().item()
            sum_vertical_velocity_sq += robot.data.root_lin_vel_b[:, 2].square().sum().item()
            support_samples += (contact_count >= 2.0).sum().item()

        return StanceMetrics(
            candidate=candidate,
            mean_height_m=sum_height / total_samples,
            support_fraction=support_samples / total_samples,
            rms_vertical_velocity_m_s=math.sqrt(sum_vertical_velocity_sq / total_samples),
            p95_height_range_m=torch.quantile(height_max - height_min, 0.95).item(),
            body_collision=body_collision,
            low_height=low_height,
        )
    finally:
        gym_env.close()


def metrics_evidence(candidate: StanceMetrics) -> dict[str, object]:
    return {
        "half_angle_rad": candidate.candidate.half_angle_rad,
        "hfe_rad": candidate.candidate.hfe_rad,
        "kfe_rad": candidate.candidate.kfe_rad,
        "spawn_height_m": candidate.candidate.spawn_height_m,
        "mean_height_m": candidate.mean_height_m,
        "support_fraction": candidate.support_fraction,
        "rms_vertical_velocity_m_s": candidate.rms_vertical_velocity_m_s,
        "p95_height_range_m": candidate.p95_height_range_m,
        "body_collision": candidate.body_collision,
        "low_height": candidate.low_height,
        "passed": passes_stance(candidate),
    }


def metrics_from_evidence(evidence: dict[str, object]) -> StanceMetrics:
    return StanceMetrics(
        candidate=StanceCandidate(
            half_angle_rad=float(evidence["half_angle_rad"]),
            hfe_rad=float(evidence["hfe_rad"]),
            kfe_rad=float(evidence["kfe_rad"]),
            spawn_height_m=float(evidence["spawn_height_m"]),
        ),
        mean_height_m=float(evidence["mean_height_m"]),
        support_fraction=float(evidence["support_fraction"]),
        rms_vertical_velocity_m_s=float(evidence["rms_vertical_velocity_m_s"]),
        p95_height_range_m=float(evidence["p95_height_range_m"]),
        body_collision=bool(evidence["body_collision"]),
        low_height=bool(evidence["low_height"]),
    )


def write_evidence(
    output: str, metrics: list[StanceMetrics], selected: StanceMetrics | None
) -> None:
    payload = {
        "target_height_m": TARGET_HEIGHT_M,
        "candidates": [metrics_evidence(candidate) for candidate in metrics],
        "selected_candidate": None
        if selected is None
        else {
            "half_angle_rad": selected.candidate.half_angle_rad,
            "hfe_rad": selected.candidate.hfe_rad,
            "kfe_rad": selected.candidate.kfe_rad,
            "spawn_height_m": selected.candidate.spawn_height_m,
        },
    }
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")


def worker_command(
    args_cli: argparse.Namespace, candidate: StanceCandidate, worker_output: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker_half_angle_rad",
        str(candidate.half_angle_rad),
        "--worker_hfe_rad",
        str(candidate.hfe_rad),
        "--worker_kfe_rad",
        str(candidate.kfe_rad),
        "--worker_spawn_height_m",
        str(candidate.spawn_height_m),
        "--worker_output",
        str(worker_output),
        "--num_envs",
        str(args_cli.num_envs),
        "--device",
        args_cli.device,
    ]
    if args_cli.headless:
        command.append("--headless")
    return command


def run_parent(args_cli: argparse.Namespace) -> int:
    if args_cli.num_envs != NUM_ENVS:
        raise ValueError(f"stance calibration requires exactly {NUM_ENVS} environments")
    metrics: list[StanceMetrics] = []
    with tempfile.TemporaryDirectory(prefix="stance-calibration-v4-") as temp_dir:
        for index, candidate in enumerate(STANCE_CANDIDATES):
            worker_output = Path(temp_dir) / f"candidate_{index}.json"
            subprocess.run(worker_command(args_cli, candidate, worker_output), check=True)
            evidence = json.loads(worker_output.read_text())
            metrics.append(metrics_from_evidence(evidence))
            print(f"[STANCE] {evidence}", flush=True)

    try:
        selected = select_stance_candidate(metrics)
    except ValueError:
        write_evidence(args_cli.output, metrics, None)
        print("[STANCE] no candidate passed", flush=True)
        return 1
    write_evidence(args_cli.output, metrics, selected)
    print(f"[STANCE] selected_candidate={selected.candidate}", flush=True)
    return 0


def worker_candidate(args_cli: argparse.Namespace) -> StanceCandidate:
    return StanceCandidate(
        half_angle_rad=args_cli.worker_half_angle_rad,
        hfe_rad=args_cli.worker_hfe_rad,
        kfe_rad=args_cli.worker_kfe_rad,
        spawn_height_m=args_cli.worker_spawn_height_m,
    )


def run_worker(args_cli: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        candidate = worker_candidate(args_cli)
        metrics = collect_candidate(candidate, args_cli)
        Path(args_cli.worker_output).write_text(
            json.dumps(metrics_evidence(metrics), indent=2) + "\n"
        )
        return 0
    finally:
        simulation_app.close()


def main() -> int:
    args_cli = parse_args()
    worker_values = (
        args_cli.worker_half_angle_rad,
        args_cli.worker_hfe_rad,
        args_cli.worker_kfe_rad,
        args_cli.worker_spawn_height_m,
        args_cli.worker_output,
    )
    worker_requested = any(value is not None for value in worker_values)
    if worker_requested:
        if any(value is None for value in worker_values):
            raise ValueError("worker mode requires all hidden worker arguments")
        return run_worker(args_cli)
    return run_parent(args_cli)


if __name__ == "__main__":
    raise SystemExit(main())
