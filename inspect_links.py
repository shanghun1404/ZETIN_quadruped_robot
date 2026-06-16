import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from for_robot_cfg import FOR_ROBOT_CFG
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from for_env_cfg import ForFlatEnvCfg

env_cfg = ForFlatEnvCfg()
env_cfg.scene.num_envs = 1
env = gym.make("Isaac-Velocity-Flat-For-v0", cfg=env_cfg)
print("Robot Link Names:", env.unwrapped.scene["robot"].data.body_names)
env.close()
simulation_app.close()
