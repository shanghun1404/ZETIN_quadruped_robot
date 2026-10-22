"""Evaluate the trained V12 depth policy in closed-loop IsaacLab rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Guide-RobotV2-v12-vision-eval"
STAIRS_APPROACH_DISTANCE_M = 2.1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--student-checkpoint", required=True)
parser.add_argument("--scenario", choices=("rough", "stairs_up", "stairs_down"), required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--terrain-level", type=int)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from evaluation_cli import load_inference_checkpoint, validate_fixed_terrain_level
from v9_env_cfg import (
    STAIRS_DOWN_EVAL_TERRAIN_CFG,
    STAIRS_NUM_LEVELS,
    STAIRS_STEP_HEIGHT_RANGE,
    STAIRS_UP_EVAL_TERRAIN_CFG,
)
from v11_terrain_env_cfg import GuideV2ScannerTerrainEnvCfg
from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG
from v11_video_geometry import stair_step_height_m
from v12_camera import camera_depth_nchw, make_v12_depth_camera_cfg
from v12_collection import policy_observation_tensor
from v12_eval_metrics import rough_gate, stair_traversal_mask, summarize_rollout
from v12_teacher_actor import load_frozen_v11_actor
from v12_training import checkpoint_is_finite
from v12_vision_student import TinyDepthViT, reconstruct_teacher_observation


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2ScannerTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": V11_TERRAIN_PPO_CFG,
    },
)


def load_student(path: str | Path, device: str) -> TinyDepthViT:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint_is_finite(checkpoint):
        raise ValueError("student checkpoint has missing or non-finite model state")
    model = TinyDepthViT()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model.to(device)


def pin_terrain(raw_env, level: int) -> None:
    base_env = raw_env.unwrapped
    terrain = base_env.scene.terrain
    generator = base_env.cfg.scene.terrain.terrain_generator
    if generator is None:
        raise ValueError("V12 evaluation requires generated terrain")
    level = validate_fixed_terrain_level(level, int(generator.num_rows))
    terrain.terrain_levels.fill_(level)
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    terrain.terrain_types[:] = env_ids % int(generator.num_cols)
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels,
        terrain.terrain_types,
    ]
    raw_env.reset()


def main() -> dict[str, object]:
    for path in (args_cli.teacher_checkpoint, args_cli.student_checkpoint):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if args_cli.num_envs <= 0 or args_cli.seconds <= 0.0:
        raise ValueError("num_envs and seconds must be positive")

    default_level = 4 if args_cli.scenario == "rough" else 5
    terrain_level = default_level if args_cli.terrain_level is None else args_cli.terrain_level
    env_cfg = GuideV2ScannerTerrainEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = max(60.0, args_cli.seconds + 1.0)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.curriculum = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        key: (0.0, 0.0)
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (60.0, 60.0)
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (args_cli.speed, args_cli.speed)
    command.ranges.ang_vel_z = (0.0, 0.0)

    if args_cli.scenario == "stairs_up":
        env_cfg.scene.terrain.terrain_generator = copy.deepcopy(
            STAIRS_UP_EVAL_TERRAIN_CFG
        )
    elif args_cli.scenario == "stairs_down":
        env_cfg.scene.terrain.terrain_generator = copy.deepcopy(
            STAIRS_DOWN_EVAL_TERRAIN_CFG
        )
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise ValueError("V12 evaluation requires generated terrain")
    generator.curriculum = False
    env_cfg.scene.terrain.max_init_terrain_level = terrain_level
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.scene.depth_camera = make_v12_depth_camera_cfg(step_dt)

    agent_cfg = copy.deepcopy(V11_TERRAIN_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    pin_terrain(raw_env, terrain_level)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        load_inference_checkpoint(runner, args_cli.teacher_checkpoint)
        runner.alg.policy.reset()
        teacher_policy = runner.get_inference_policy(device=agent_cfg.device)
        frozen_actor = load_frozen_v11_actor(
            args_cli.teacher_checkpoint,
            device=agent_cfg.device,
        )
        student = load_student(args_cli.student_checkpoint, agent_cfg.device)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        camera = base_env.scene["depth_camera"]

        observations = env.get_observations()
        policy_observation = policy_observation_tensor(observations)
        with torch.inference_mode():
            actor_equivalence_error = float(
                (teacher_policy(observations) - frozen_actor(policy_observation))
                .abs()
                .max()
            )
        if actor_equivalence_error > 1.0e-6:
            raise RuntimeError(
                f"manual actor equivalence failed: {actor_equivalence_error}"
            )

        steps = int(args_cli.seconds / base_env.step_dt)
        alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device=base_env.device)
        start_xy = robot.data.root_pos_w[:, :2].clone()
        start_height = robot.data.root_pos_w[:, 2].clone()
        max_progress = torch.zeros(args_cli.num_envs, device=base_env.device)
        vertical_after_approach = torch.zeros(
            args_cli.num_envs,
            device=base_env.device,
        )
        valid_history = []
        speed_history = []
        scan_error_history = []
        action_error_history = []

        with torch.inference_mode():
            for _ in range(steps):
                policy_observation = policy_observation_tensor(observations)
                depth = camera_depth_nchw(camera)
                predicted_scan = student(depth)
                student_observation = reconstruct_teacher_observation(
                    policy_observation[:, :48],
                    predicted_scan,
                )
                student_action = frozen_actor(student_observation)
                teacher_action = teacher_policy(observations)

                progress = torch.linalg.vector_norm(
                    robot.data.root_pos_w[:, :2] - start_xy,
                    dim=-1,
                )
                height_gain = robot.data.root_pos_w[:, 2] - start_height
                max_progress = torch.where(
                    alive,
                    torch.maximum(max_progress, progress),
                    max_progress,
                )
                reached = alive & (progress >= STAIRS_APPROACH_DISTANCE_M)
                vertical_value = (
                    -height_gain
                    if args_cli.scenario == "stairs_down"
                    else height_gain
                )
                vertical_after_approach = torch.where(
                    reached,
                    torch.maximum(vertical_after_approach, vertical_value),
                    vertical_after_approach,
                )
                valid_history.append(alive.cpu())
                speed_history.append(robot.data.root_lin_vel_b[:, 1].cpu())
                scan_error_history.append(
                    (predicted_scan - policy_observation[:, 48:])
                    .abs()
                    .mean(dim=-1)
                    .cpu()
                )
                action_error_history.append(
                    (student_action - teacher_action).abs().mean(dim=-1).cpu()
                )

                observations, _, dones, _ = env.step(student_action)
                runner.alg.policy.reset(dones)
                alive &= ~dones.bool()

        summary = summarize_rollout(
            torch.stack(valid_history),
            alive.cpu(),
            torch.stack(speed_history),
            torch.stack(scan_error_history),
            torch.stack(action_error_history),
        )
        result = {
            "scenario": args_cli.scenario,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "seconds": args_cli.seconds,
            "speed_command_m_s": args_cli.speed,
            "terrain_level": terrain_level,
            "actor_equivalence_max_abs_error": actor_equivalence_error,
            **summary,
            "max_progress_mean_m": float(max_progress.mean()),
        }
        if args_cli.scenario == "rough":
            result["operational_pass"] = rough_gate(
                summary["survival"],
                summary["mean_forward_speed_m_s"],
            )
        else:
            step_height = stair_step_height_m(
                STAIRS_STEP_HEIGHT_RANGE,
                STAIRS_NUM_LEVELS,
                terrain_level,
            )
            traversed = stair_traversal_mask(
                args_cli.scenario,
                alive,
                max_progress,
                vertical_after_approach,
                approach_distance_m=STAIRS_APPROACH_DISTANCE_M,
                step_height_m=step_height,
            )
            traversal_rate = float(traversed.float().mean())
            result.update(
                {
                    "step_height_m": step_height,
                    "vertical_after_approach_mean_m": float(
                        vertical_after_approach.mean()
                    ),
                    "traversed_count": int(traversed.sum()),
                    "traversal_rate": traversal_rate,
                    "operational_pass": (
                        summary["survival"] >= 0.80 and traversal_rate >= 0.50
                    ),
                }
            )

        output = Path(args_cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
