"""Record the selected V11 scanner policy crossing a physical stair row."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Stairs-Guide-RobotV2-Video-v11"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", default="output_video_v11_stairs")
parser.add_argument("--view", choices=("side", "rear_quarter"), default="side")
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--video-length", type=int, default=600)
parser.add_argument("--stair-level", type=int, default=5)
parser.add_argument("--seed", type=int, default=46)
parser.add_argument(
    "--scenario",
    choices=("stairs_up", "stairs_down"),
    default="stairs_up",
)
parser.add_argument(
    "--probe-only",
    action="store_true",
    help="run identical physics without camera rendering or MP4 encoding",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.probe_only
args_cli.headless = True
args_cli.width = 640
args_cli.height = 480
args_cli.kit_args = " ".join(
    (
        "--/rtx/sceneDb/maxMemoryBudget=512",
        "--/renderer/multiGpu/enabled=false",
        "--/rtx/raytracing/enabled=false",
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
from evaluation_cli import load_inference_checkpoint
from v9_env_cfg import (
    STAIRS_DOWN_EVAL_TERRAIN_CFG,
    STAIRS_NUM_LEVELS,
    STAIRS_STEP_HEIGHT_RANGE,
    STAIRS_STEP_WIDTH,
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
from video_artifacts import copy_exact_recording, create_fresh_recording_dir


STAIRS_PLATFORM_WIDTH_M = 3.0
STAIRS_APPROACH_DISTANCE_M = (
    STAIRS_PLATFORM_WIDTH_M / 2.0 + 2.0 * STAIRS_STEP_WIDTH
)


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2ScannerTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": V11_TERRAIN_PPO_CFG,
    },
)


def _set_camera(env, view: str) -> None:
    root = env.unwrapped.scene["robot"].data.root_pos_w[0]
    root_xyz = tuple(float(value) for value in root)
    eye, target = tracking_camera_pose(root_xyz, view)
    env.unwrapped.sim.set_camera_view(eye=eye, target=target)


def _create_head_marker() -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/v11_head_marker",
            markers={
                "head": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 1.0, 0.10)
                    ),
                )
            },
        )
    )


def _update_head_marker(env, marker: VisualizationMarkers) -> None:
    robot = env.unwrapped.scene["robot"]
    offset = torch.tensor(
        [[0.0, 0.30, 0.05]],
        device=robot.data.root_pos_w.device,
    )
    position = robot.data.root_pos_w[:1] + quat_apply(
        robot.data.root_quat_w[:1],
        offset,
    )
    marker.visualize(position)


def _pin_stair_level(raw_env, level: int) -> None:
    terrain = raw_env.unwrapped.scene.terrain
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels,
        terrain.terrain_types,
    ]
    raw_env.reset()


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


def main() -> None:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.video_length <= 0:
        raise ValueError("video length must be positive")
    step_height_m = stair_step_height_m(
        STAIRS_STEP_HEIGHT_RANGE,
        STAIRS_NUM_LEVELS,
        args_cli.stair_level,
    )

    env_cfg = GuideV2ScannerTerrainEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 60.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.terrain.terrain_generator = (
        STAIRS_DOWN_EVAL_TERRAIN_CFG
        if args_cli.scenario == "stairs_down"
        else STAIRS_UP_EVAL_TERRAIN_CFG
    )
    env_cfg.scene.terrain.max_init_terrain_level = args_cli.stair_level
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
    command.debug_vis = False

    agent_cfg = copy.deepcopy(V11_TERRAIN_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device

    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(args_cli.checkpoint).stem
    height_mm = round(step_height_m * 1000)
    prefix = (
        f"v11_{checkpoint_name}_{args_cli.scenario}_level{args_cli.stair_level}_"
        f"{height_mm}mm_{args_cli.view}_speed{round(args_cli.speed * 100):03d}_"
        f"seed{args_cli.seed}"
    )
    final_path = output_dir / f"{prefix}.mp4"
    telemetry_path = output_dir / f"{prefix}.json"

    raw_env = gym.make(
        TASK_ID,
        cfg=env_cfg,
        render_mode=None if args_cli.probe_only else "rgb_array",
    )
    _pin_stair_level(raw_env, args_cli.stair_level)
    env = raw_env
    record_dir = None
    if not args_cli.probe_only:
        record_dir = create_fresh_recording_dir(output_dir, prefix)
        env = TrackingWrapper(env, args_cli.view)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=record_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix=prefix,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    alive = True
    peak_height_after_approach_m = float("-inf")
    max_descent_after_approach_m = float("-inf")
    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        load_inference_checkpoint(runner, args_cli.checkpoint)
        runner.alg.policy.reset()
        policy = runner.get_inference_policy(device=agent_cfg.device)
        robot = env.unwrapped.scene["robot"]
        start_xy = robot.data.root_pos_w[0, :2].clone()
        start_height = float(robot.data.root_pos_w[0, 2])
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        with torch.inference_mode():
            for _ in range(args_cli.video_length):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                runner.alg.policy.reset(dones)
                progress_m = float(
                    torch.linalg.vector_norm(
                        robot.data.root_pos_w[0, :2] - start_xy
                    )
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
    finally:
        env.close()

    final_progress_m = float(
        torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - start_xy)
    )
    if peak_height_after_approach_m == float("-inf"):
        peak_height_after_approach_m = float(robot.data.root_pos_w[0, 2]) - start_height
    if max_descent_after_approach_m == float("-inf"):
        max_descent_after_approach_m = start_height - float(
            robot.data.root_pos_w[0, 2]
        )
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
    if record_dir is not None:
        copy_exact_recording(record_dir, prefix, final_path)
    telemetry = {
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "seed": args_cli.seed,
        "scenario": args_cli.scenario,
        "stair_level": args_cli.stair_level,
        "step_height_m": step_height_m,
        "view": args_cli.view,
        "speed_m_s": args_cli.speed,
        "video_frames": args_cli.video_length,
        "alive_at_end": alive,
        "final_progress_m": final_progress_m,
        "peak_height_after_approach_m": peak_height_after_approach_m,
        "max_descent_after_approach_m": max_descent_after_approach_m,
        "climbed": traversed if args_cli.scenario == "stairs_up" else False,
        "descended": traversed if args_cli.scenario == "stairs_down" else False,
        "traversed": traversed,
        "video": None if args_cli.probe_only else str(final_path),
    }
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n")
    if not args_cli.probe_only:
        print(f"[VIDEO] {final_path}")
    print(f"[TELEMETRY] {telemetry_path}")
    print(f"[TRAVERSED] {traversed}")


if __name__ == "__main__":
    main()
    simulation_app.close()
