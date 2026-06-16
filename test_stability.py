import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from for_env_cfg import ForFlatEnvCfg

cfg = ForFlatEnvCfg()
cfg.scene.num_envs = 2
env = ManagerBasedEnv(cfg=cfg)

obs, _ = env.reset()
for i in range(50):
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    obs, _, _, _, _ = env.step(actions)
    pos = env.scene["robot"].data.root_pos_w
    vel = env.scene["robot"].data.root_lin_vel_w
    print(f"Step {i}: Pos: {pos[0].cpu().numpy()}, Vel: {vel[0].cpu().numpy()}")

simulation_app.close()
