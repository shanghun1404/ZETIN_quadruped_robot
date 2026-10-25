"""Fine-tune the V11 actor for V14 physical inter-leg collision avoidance."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Guide-RobotV2-v14-leg-collision"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--actor-checkpoint", required=True)
parser.add_argument("--student-checkpoint", required=True)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initial-terrain-level", type=int, default=0)
parser.add_argument("--stair-focus", action="store_true")
parser.add_argument("--direction-focus", action="store_true")
parser.add_argument("--yaw-tracking-std", type=float, default=0.25)
parser.add_argument("--yaw-reward-weight", type=float, default=7.5)
parser.add_argument("--direction-learning-rate", type=float, default=3.0e-4)
parser.add_argument("--output-root", default="logs/rsl_rl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner

from training_resume import validate_initial_terrain_level
from v13_depth_policy import make_v13_depth_wrapper
from v14_leg_env_cfg import GuideV14LegEnvCfg, GuideV14StairEnvCfg, GuideV14DirectionEnvCfg, V14_REWARD_WEIGHTS
from v14_leg_ppo_cfg import V14_LEG_PPO_CFG


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV14LegEnvCfg,
        "rsl_rl_cfg_entry_point": V14_LEG_PPO_CFG,
    },
)


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
    if args_cli.num_envs <= 0 or args_cli.max_iterations <= 0:
        raise ValueError("num_envs and max_iterations must be positive")
    if args_cli.direction_focus and not (
        0.05 <= args_cli.yaw_tracking_std <= 2.0 and 0.0 < args_cli.yaw_reward_weight <= 10.0
        and 0.0 < args_cli.direction_learning_rate <= 1.0e-3
    ):
        raise ValueError("invalid bounded direction-adaptation parameters")

    if args_cli.direction_focus and args_cli.stair_focus:
        raise ValueError("choose direction-focus or stair-focus")
    env_cfg = (GuideV14DirectionEnvCfg() if args_cli.direction_focus else
               GuideV14StairEnvCfg() if args_cli.stair_focus else GuideV14LegEnvCfg())
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
    # V12 student data collection and closed-loop evaluation use clean proprioception.
    env_cfg.observations.policy.enable_corruption = False
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise ValueError("V14 requires V12's generated terrain")
    level = validate_initial_terrain_level(
        args_cli.initial_terrain_level, int(generator.num_rows)
    )
    env_cfg.scene.terrain.max_init_terrain_level = level
    if args_cli.direction_focus:
        # One fixed terrain column per robot prevents visually overlapping peers.
        generator.num_cols = args_cli.num_envs
        env_cfg.rewards.base_angular_velocity.params["std"] = args_cli.yaw_tracking_std
        env_cfg.rewards.base_angular_velocity.weight = args_cli.yaw_reward_weight
    env_cfg.seed = args_cli.seed

    agent_cfg = copy.deepcopy(V14_LEG_PPO_CFG)
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.direction_focus:
        agent_cfg.experiment_name = "guide_v14_direction_adaptation"
        agent_cfg.run_name = "v14_m5700_lane_yaw"
        agent_cfg.save_interval = 100
        agent_cfg.algorithm.learning_rate = args_cli.direction_learning_rate
        agent_cfg.algorithm.schedule = "fixed"
    if args_cli.stair_focus:
        agent_cfg.experiment_name = "guide_v14_stair_adaptation"
        agent_cfg.run_name = "v14_m5700_stair_focus"
    log_root = Path(args_cli.output_root).resolve() / agent_cfg.experiment_name
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] V14 log directory: {log_dir}", flush=True)

    manifest = {
        "actor_checkpoint": str(Path(args_cli.actor_checkpoint).resolve()),
        "actor_sha256": file_sha256(args_cli.actor_checkpoint),
        "frozen_v12_student": str(Path(args_cli.student_checkpoint).resolve()),
        "student_sha256": file_sha256(args_cli.student_checkpoint),
        "usd_path": str(Path(args_cli.usd_path).resolve()),
        "usd_sha256": file_sha256(args_cli.usd_path),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "initial_terrain_level": level,
        "stair_focus": args_cli.stair_focus,
        "direction_focus": args_cli.direction_focus,
        "terrain_columns": generator.num_cols,
        "yaw_tracking_std": env_cfg.rewards.base_angular_velocity.params["std"],
        "learning_rate": agent_cfg.algorithm.learning_rate,
        "learning_rate_schedule": agent_cfg.algorithm.schedule,
        "terrain_proportions": {
            name: float(terrain.proportion)
            for name, terrain in generator.sub_terrains.items()
        },
        "reward_weights": {**V14_REWARD_WEIGHTS,
                           "base_angular_velocity": env_cfg.rewards.base_angular_velocity.weight},
        "action_lpf_hz": 25.0,
        "effort_guard_fraction": 0.99,
        "enabled_self_collisions": True,
        "leg_terrain_filtered": True,
        "replicate_physics": False,
        "control_dt_s": env_cfg.decimation * env_cfg.sim.dt,
        "filter_dt_s": env_cfg.sim.dt,
    }
    (log_dir / "v14_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("[INFO] Constructing V14 IsaacLab environment", flush=True)
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    print("[INFO] Environment constructed", flush=True)
    DepthWrapper = make_v13_depth_wrapper()
    env = DepthWrapper(
        raw_env,
        student_checkpoint=args_cli.student_checkpoint,
        clip_actions=agent_cfg.clip_actions,
    )
    if args_cli.direction_focus:
        import torch

        obs = env.get_observations()["policy"]
        origins = env.unwrapped.scene.terrain.env_origins
        if obs.shape != (args_cli.num_envs, 235) or not bool(torch.isfinite(obs).all()):
            raise RuntimeError("direction adaptation requires finite 235-dimensional observations")
        if len(torch.unique(origins, dim=0)) != args_cli.num_envs:
            raise RuntimeError("direction training robots must have unique terrain origins")
        print(f"[INFO] Direction smoke: obs={tuple(obs.shape)}, unique origins={len(torch.unique(origins, dim=0))}", flush=True)
    print("[INFO] Frozen V12 depth wrapper constructed", flush=True)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        print("[INFO] PPO runner constructed", flush=True)
        runner.load(args_cli.actor_checkpoint, load_optimizer=False, map_location=agent_cfg.device)
        print("[INFO] initial actor checkpoint loaded", flush=True)
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        print("[INFO] PPO iterations finished", flush=True)
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
