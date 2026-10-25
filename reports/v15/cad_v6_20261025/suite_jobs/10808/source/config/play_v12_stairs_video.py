"""Record a closed-loop V12 depth-ViT student on deterministic stairs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Stairs-Guide-RobotV2-Video-v12"
STAIRS_APPROACH_DISTANCE_M = 2.1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--student-checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--view", choices=("side", "rear_quarter"), default="side")
parser.add_argument("--scenario", choices=("stairs_up", "stairs_down"), default="stairs_up")
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--video-length", type=int, default=600)
parser.add_argument("--stair-level", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
args_cli.width = 640
args_cli.height = 480
args_cli.kit_args = " ".join(
    (
        "--/rtx/sceneDb/maxMemoryBudget=512",
        "--/renderer/multiGpu/enabled=false",
    )
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from evaluation_cli import load_inference_checkpoint, validate_fixed_terrain_level
from v9_env_cfg import (
    STAIRS_DOWN_EVAL_TERRAIN_CFG,
    STAIRS_NUM_LEVELS,
    STAIRS_STEP_HEIGHT_RANGE,
    STAIRS_UP_EVAL_TERRAIN_CFG,
)
from v11_terrain_env_cfg import GuideV2ScannerTerrainEnvCfg
from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG
from v11_video_geometry import (
    stair_descent_passed,
    stair_rollout_passed,
    stair_step_height_m,
    tracking_camera_pose,
)
from v12_camera import camera_depth_nchw, make_v12_depth_camera_cfg
from v12_collection import policy_observation_tensor
from v12_teacher_actor import load_frozen_v11_actor
from v12_training import checkpoint_is_finite
from v12_vision_student import TinyDepthViT, reconstruct_teacher_observation
from video_artifacts import copy_exact_recording, create_fresh_recording_dir


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2ScannerTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": V11_TERRAIN_PPO_CFG,
    },
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("V12 video requires generated terrain")
    fixed_level = validate_fixed_terrain_level(level, int(generator.num_rows))
    terrain.terrain_levels.fill_(fixed_level)
    terrain.terrain_types.zero_()
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels,
        terrain.terrain_types,
    ]
    raw_env.reset()


def _set_camera(env, view: str) -> None:
    root = env.unwrapped.scene["robot"].data.root_pos_w[0]
    eye, target = tracking_camera_pose(tuple(float(value) for value in root), view)
    env.unwrapped.sim.set_camera_view(eye=eye, target=target)


def _create_head_marker() -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/v12_head_marker",
            markers={
                "head": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.45, 1.0)
                    ),
                )
            },
        )
    )


def _update_head_marker(env, marker: VisualizationMarkers) -> None:
    robot = env.unwrapped.scene["robot"]
    offset = torch.tensor([[0.0, 0.30, 0.05]], device=robot.data.root_pos_w.device)
    position = robot.data.root_pos_w[:1] + quat_apply(robot.data.root_quat_w[:1], offset)
    marker.visualize(position)


class TrackingWrapper(gym.Wrapper):
    def __init__(self, env, view: str):
        super().__init__(env)
        self.view = view
        self.marker = _create_head_marker()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        _set_camera(self.env, self.view)
        _update_head_marker(self.env, self.marker)
        return result

    def step(self, action):
        result = self.env.step(action)
        _set_camera(self.env, self.view)
        _update_head_marker(self.env, self.marker)
        return result


def main() -> dict[str, object]:
    for path in (args_cli.teacher_checkpoint, args_cli.student_checkpoint):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if args_cli.video_length <= 0:
        raise ValueError("video length must be positive")
    if args_cli.speed <= 0.0:
        raise ValueError("speed must be positive")

    env_cfg = GuideV2ScannerTerrainEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = max(60.0, args_cli.video_length * 0.02 + 1.0)
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
    env_cfg.scene.terrain.terrain_generator = copy.deepcopy(
        STAIRS_DOWN_EVAL_TERRAIN_CFG
        if args_cli.scenario == "stairs_down"
        else STAIRS_UP_EVAL_TERRAIN_CFG
    )
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise ValueError("V12 video requires generated terrain")
    generator.curriculum = False
    env_cfg.scene.terrain.max_init_terrain_level = args_cli.stair_level
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (60.0, 60.0)
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (args_cli.speed, args_cli.speed)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.debug_vis = False
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.scene.depth_camera = make_v12_depth_camera_cfg(step_dt)

    agent_cfg = copy.deepcopy(V11_TERRAIN_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device
    step_height_m = stair_step_height_m(
        STAIRS_STEP_HEIGHT_RANGE,
        STAIRS_NUM_LEVELS,
        args_cli.stair_level,
    )
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    student_name = Path(args_cli.student_checkpoint).stem
    prefix = (
        f"v12_{student_name}_{args_cli.scenario}_level{args_cli.stair_level}_"
        f"{round(step_height_m * 1000)}mm_{args_cli.view}_"
        f"speed{round(args_cli.speed * 100):03d}_seed{args_cli.seed}"
    )
    final_path = output_dir / f"{prefix}.mp4"
    telemetry_path = output_dir / f"{prefix}.json"
    record_dir = create_fresh_recording_dir(output_dir, prefix)

    raw_env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array")
    pin_terrain(raw_env, args_cli.stair_level)
    env = TrackingWrapper(raw_env, args_cli.view)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=record_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        name_prefix=prefix,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    actor_equivalence_error = float("nan")
    scan_mae_sum = 0.0
    action_mae_sum = 0.0
    alive = True
    final_progress_m = 0.0
    peak_height_after_approach_m = float("-inf")
    max_descent_after_approach_m = float("-inf")
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        load_inference_checkpoint(runner, args_cli.teacher_checkpoint)
        runner.alg.policy.reset()
        teacher_policy = runner.get_inference_policy(device=agent_cfg.device)
        frozen_actor = load_frozen_v11_actor(
            args_cli.teacher_checkpoint,
            device=agent_cfg.device,
        )
        student = load_student(args_cli.student_checkpoint, agent_cfg.device)
        robot = env.unwrapped.scene["robot"]
        depth_camera = env.unwrapped.scene["depth_camera"]
        observations = env.get_observations()
        policy_observation = policy_observation_tensor(observations)
        with torch.no_grad():
            actor_equivalence_error = float(
                (teacher_policy(observations) - frozen_actor(policy_observation))
                .abs()
                .max()
            )
        if actor_equivalence_error > 1.0e-6:
            raise RuntimeError(
                f"manual actor equivalence failed: {actor_equivalence_error}"
            )

        start_xy = robot.data.root_pos_w[0, :2].clone()
        start_height = float(robot.data.root_pos_w[0, 2])
        with torch.no_grad():
            for _ in range(args_cli.video_length):
                policy_observation = policy_observation_tensor(observations)
                depth = camera_depth_nchw(depth_camera)
                predicted_scan = student(depth)
                student_observation = reconstruct_teacher_observation(
                    policy_observation[:, :48],
                    predicted_scan,
                )
                student_action = frozen_actor(student_observation)
                teacher_action = teacher_policy(observations)
                scan_mae_sum += float(
                    (predicted_scan - policy_observation[:, 48:]).abs().mean()
                )
                action_mae_sum += float((student_action - teacher_action).abs().mean())
                observations, _, dones, _ = env.step(student_action)
                runner.alg.policy.reset(dones)
                progress_m = float(
                    torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - start_xy)
                )
                height_gain_m = float(robot.data.root_pos_w[0, 2]) - start_height
                if alive and progress_m >= STAIRS_APPROACH_DISTANCE_M:
                    peak_height_after_approach_m = max(
                        peak_height_after_approach_m,
                        height_gain_m,
                    )
                    max_descent_after_approach_m = max(
                        max_descent_after_approach_m,
                        -height_gain_m,
                    )
                alive = alive and not bool(dones[0])

        final_progress_m = float(
            torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - start_xy)
        )
        if peak_height_after_approach_m == float("-inf"):
            peak_height_after_approach_m = float(robot.data.root_pos_w[0, 2]) - start_height
        if max_descent_after_approach_m == float("-inf"):
            max_descent_after_approach_m = start_height - float(robot.data.root_pos_w[0, 2])
    finally:
        env.close()

    copy_exact_recording(record_dir, prefix, final_path)
    if args_cli.scenario == "stairs_down":
        traversed = stair_descent_passed(
            alive=alive,
            final_progress_m=final_progress_m,
            max_descent_after_approach_m=max_descent_after_approach_m,
            approach_distance_m=STAIRS_APPROACH_DISTANCE_M,
            step_height_m=step_height_m,
        )
    else:
        traversed = stair_rollout_passed(
            alive=alive,
            final_progress_m=final_progress_m,
            peak_height_after_approach_m=peak_height_after_approach_m,
            approach_distance_m=STAIRS_APPROACH_DISTANCE_M,
            step_height_m=step_height_m,
        )
    telemetry = {
        "policy_version": "V12",
        "policy_path": "depth_to_vit_to_frozen_v11_actor",
        "teacher_checkpoint": str(Path(args_cli.teacher_checkpoint).resolve()),
        "teacher_checkpoint_sha256": sha256_file(args_cli.teacher_checkpoint),
        "student_checkpoint": str(Path(args_cli.student_checkpoint).resolve()),
        "student_checkpoint_sha256": sha256_file(args_cli.student_checkpoint),
        "scenario": args_cli.scenario,
        "seed": args_cli.seed,
        "stair_level": args_cli.stair_level,
        "step_height_m": step_height_m,
        "view": args_cli.view,
        "speed_m_s": args_cli.speed,
        "video_frames": args_cli.video_length,
        "alive_at_end": alive,
        "traversed": traversed,
        "final_progress_m": final_progress_m,
        "peak_height_after_approach_m": peak_height_after_approach_m,
        "max_descent_after_approach_m": max_descent_after_approach_m,
        "actor_equivalence_max_abs_error": actor_equivalence_error,
        "mean_scan_mae": scan_mae_sum / args_cli.video_length,
        "mean_action_mae": action_mae_sum / args_cli.video_length,
        "video": str(final_path),
        "video_sha256": sha256_file(final_path),
    }
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(telemetry, indent=2), flush=True)
    return telemetry


if __name__ == "__main__":
    main()
    simulation_app.close()
