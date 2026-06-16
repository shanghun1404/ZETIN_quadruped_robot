import argparse
import sys
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-For-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training.")
parser.add_argument("--load_run", type=str, default=".*", help="Name of the run to load when resuming.")
parser.add_argument("--checkpoint", type=str, default=".*", help="Saved model checkpoint number.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import logging
from datetime import datetime

import importlib.metadata as metadata
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

# Import custom configs
from for_env_cfg import ForFlatEnvCfg, ForRoughEnvCfg
from for_ppo_cfg import ForFlatPPORunnerCfg, ForRoughPPORunnerCfg

try:
    installed_version = metadata.version("rsl-rl-lib")
except Exception:
    installed_version = "3.0.0"



##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-For-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForFlatEnvCfg,
        "rsl_rl_cfg_entry_point": ForFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-For-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForRoughEnvCfg,
        "rsl_rl_cfg_entry_point": ForRoughPPORunnerCfg,
    },
)

def main():
    # task and agent configs
    if args_cli.task == "Isaac-Velocity-Flat-For-v0":
        env_cfg = ForFlatEnvCfg()
        agent_cfg = ForFlatPPORunnerCfg()
    else:
        env_cfg = ForRoughEnvCfg()
        agent_cfg = ForRoughPPORunnerCfg()

    # override configurations
    # 로봇개 조사자료 반영: 2048대 스폰 + 미니배치 64분할로 메모리 확보
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 2048
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    agent_cfg.resume = args_cli.resume
    agent_cfg.load_run = args_cli.load_run
    agent_cfg.load_checkpoint = args_cli.checkpoint

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    agent_cfg.seed = env_cfg.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    agent_cfg_dict = agent_cfg.to_dict()
    if "encoder" not in agent_cfg_dict:
        agent_cfg_dict["encoder"] = {}
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # resume training
    if agent_cfg.resume:
        from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
