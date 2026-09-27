"""Large-scale V15 fine-tuning of the V14 5700 actor on PhysX joint-drive PD."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Guide-RobotV2-v15-locomotion"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--actor-checkpoint", required=True, help="V14 5700 actor, or a V15 checkpoint with --resume")
parser.add_argument("--student-checkpoint", required=True)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--resume", action="store_true", help="Continue a V15 run: restore optimizer and iteration")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=6000)
parser.add_argument("--seed", type=int, default=47)
parser.add_argument("--terrain-columns", type=int, help="Default: one tile per ~4 robots so depth images rarely see peers")
parser.add_argument("--max-delay-steps", type=int, default=3)
parser.add_argument("--output-root", default="logs/rsl_rl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner

from v13_depth_policy import make_v13_depth_wrapper
from v14_leg_ppo_cfg import V14_LEG_PPO_CFG
from v15_env_cfg import V15_GAIN_SCALE_RANGE, GuideV15LocomotionEnvCfg


gym.register(id=TASK_ID, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path in (args_cli.actor_checkpoint, args_cli.student_checkpoint, args_cli.usd_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if args_cli.num_envs <= 0 or args_cli.max_iterations <= 0 or not 0 <= args_cli.max_delay_steps <= 10:
        raise ValueError("num_envs, max_iterations and max-delay-steps must be valid")

    env_cfg = GuideV15LocomotionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
    env_cfg.actions.joint_pos.max_delay_steps = args_cli.max_delay_steps
    # V12 student data collection and closed-loop evaluation use clean proprioception.
    env_cfg.observations.policy.enable_corruption = False
    generator = env_cfg.scene.terrain.terrain_generator
    generator.num_cols = args_cli.terrain_columns or max(20, math.ceil(args_cli.num_envs / 4))
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.seed = args_cli.seed

    agent_cfg = copy.deepcopy(V14_LEG_PPO_CFG)
    agent_cfg.experiment_name = "guide_v15_physx_pd"
    agent_cfg.run_name = "v14_m5700_physx_pd_lane_stairs"
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = 100
    log_dir = (Path(args_cli.output_root).resolve() / agent_cfg.experiment_name
               / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"))
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] V15 log directory: {log_dir}", flush=True)

    manifest = {
        "schema": 1,
        "initial_checkpoint": str(Path(args_cli.actor_checkpoint).resolve()),
        "initial_checkpoint_sha256": file_sha256(args_cli.actor_checkpoint),
        "resume": args_cli.resume,
        "frozen_v12_student": str(Path(args_cli.student_checkpoint).resolve()),
        "student_sha256": file_sha256(args_cli.student_checkpoint),
        "usd_path": str(Path(args_cli.usd_path).resolve()),
        "usd_sha256": file_sha256(args_cli.usd_path),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "terrain_rows": generator.num_rows,
        "terrain_columns": generator.num_cols,
        "terrain_proportions": {name: float(t.proportion) for name, t in generator.sub_terrains.items()},
        "joint_drive": "physx_implicit_pd",
        "pd_gain_scale_range": list(V15_GAIN_SCALE_RANGE),
        "max_action_delay_steps": args_cli.max_delay_steps,
        "action_lpf_hz": 25.0,
        "effort_guard_fraction": env_cfg.actions.joint_pos.effort_guard_fraction,
        "lane_heading_command": True,
        "base_angular_velocity_reward": {"weight": env_cfg.rewards.base_angular_velocity.weight,
                                         "std": env_cfg.rewards.base_angular_velocity.params.get("std")},
        "learning_rate": agent_cfg.algorithm.learning_rate,
        "learning_rate_schedule": agent_cfg.algorithm.schedule,
        "control_dt_s": env_cfg.decimation * env_cfg.sim.dt,
        "physics_dt_s": env_cfg.sim.dt,
    }
    (log_dir / "v15_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    env = make_v13_depth_wrapper()(raw_env, student_checkpoint=args_cli.student_checkpoint,
                                   clip_actions=agent_cfg.clip_actions)
    obs = env.get_observations()["policy"]
    if obs.shape != (args_cli.num_envs, 235) or not bool(torch.isfinite(obs).all()):
        raise RuntimeError("V15 requires finite 235-dimensional depth observations")
    stiffness = raw_env.unwrapped.scene["robot"].data.joint_stiffness
    print(f"[INFO] V15 obs={tuple(obs.shape)} stiffness range=({float(stiffness.min()):.1f}, "
          f"{float(stiffness.max()):.1f}) terrain={generator.num_rows}x{generator.num_cols}", flush=True)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        runner.load(args_cli.actor_checkpoint, load_optimizer=args_cli.resume, map_location=agent_cfg.device)
        if args_cli.resume:
            runner.current_learning_iteration += 1
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print("[INFO] V15 PPO iterations finished", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # Kit fast shutdown must preserve the failed job's exit status.
        import omni.kit.app

        omni.kit.app.get_app().post_uncancellable_quit(1)
        simulation_app.close()
        raise SystemExit(1)
    simulation_app.close()
