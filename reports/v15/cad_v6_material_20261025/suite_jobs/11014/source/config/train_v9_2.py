import argparse
import copy
import os
from datetime import datetime

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Rough-Guide-Terrain-v9-2"

parser = argparse.ArgumentParser(description="Train Stage 1 V9.2: V9.1 with a curriculum-compatible yaw band.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=8000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner

from v9_2_env_cfg import GuideTerrainTurnBandEnvCfg
from v9_2_ppo_cfg import V9_2_PPO_CFG


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideTerrainTurnBandEnvCfg,
        "rsl_rl_cfg_entry_point": V9_2_PPO_CFG,
    },
)


def main() -> None:
    env_cfg = GuideTerrainTurnBandEnvCfg()
    agent_cfg = copy.deepcopy(V9_2_PPO_CFG)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    agent_cfg.seed = env_cfg.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_name)
    print(f"[INFO] Log directory: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
