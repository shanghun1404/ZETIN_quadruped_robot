"""Train V11 scanner terrain from an expanded V10.4 checkpoint."""

from __future__ import annotations

import argparse
import copy
import os
from datetime import datetime

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Rough-Guide-RobotV2-v11-scanner-terrain"
DEFAULT_BOOTSTRAP = "bootstrap/v11_exact_resume_from_v10_4_model_2000.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=4000)
parser.add_argument("--checkpoint", default=DEFAULT_BOOTSTRAP)
parser.add_argument(
    "--initial-terrain-level",
    type=int,
    default=0,
    help="exact starting row; the terrain curriculum remains active after reset",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner

from v11_terrain_env_cfg import GuideV2ScannerTerrainEnvCfg
from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG
from training_resume import load_training_checkpoint, validate_initial_terrain_level


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2ScannerTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": V11_TERRAIN_PPO_CFG,
    },
)


def main() -> None:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    env_cfg = GuideV2ScannerTerrainEnvCfg()
    agent_cfg = copy.deepcopy(V11_TERRAIN_PPO_CFG)
    env_cfg.scene.num_envs = args_cli.num_envs
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise ValueError("V11 training requires generated terrain")
    initial_terrain_level = validate_initial_terrain_level(
        args_cli.initial_terrain_level,
        int(generator.num_rows),
    )
    env_cfg.scene.terrain.max_init_terrain_level = initial_terrain_level
    agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    agent_cfg.seed = env_cfg.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_name)
    print(f"[INFO] Log directory: {log_dir}")

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if initial_terrain_level > 0:
        terrain = raw_env.unwrapped.scene.terrain
        terrain.terrain_levels.fill_(initial_terrain_level)
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels,
            terrain.terrain_types,
        ]
        raw_env.reset()
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )
        load_training_checkpoint(
            runner,
            args_cli.checkpoint,
            map_location=agent_cfg.device,
        )
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
