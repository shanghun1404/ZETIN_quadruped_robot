"""Record a V9 checkpoint rollout for visual gait comparison."""

import argparse
import copy
import os

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Rough-Guide-Terrain-Video-v9"

parser = argparse.ArgumentParser(description="Record a V9 terrain rollout.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--view", choices=("side", "rear_quarter"), required=True)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--output_dir", default="output_video_v9")
parser.add_argument("--scenario", choices=("rough", "stairs_up"), default="rough")
# The recording must run under the same termination set the checkpoint trained
# with, or the robot ends episodes on contacts its own policy learned to allow.
parser.add_argument("--env", choices=("v9", "v9_1", "v9_2", "v9_3"), default="v9_3")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
args_cli.width = 640
args_cli.height = 480
args_cli.kit_args = " ".join((
    "--/rtx/sceneDb/maxMemoryBudget=512",
    "--/renderer/multiGpu/enabled=false",
    "--/rtx/raytracing/enabled=false",
))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from v9_env_cfg import GuideTerrainEnvCfg, STAIRS_NUM_LEVELS, STAIRS_UP_EVAL_TERRAIN_CFG
from v9_ppo_cfg import V9_PPO_CFG

if args_cli.env == "v9_3":
    from v9_3_env_cfg import GuideTerrainSharpVelEnvCfg as VIDEO_ENV_CFG
elif args_cli.env == "v9_2":
    from v9_2_env_cfg import GuideTerrainTurnBandEnvCfg as VIDEO_ENV_CFG
elif args_cli.env == "v9_1":
    from v9_1_env_cfg import GuideTerrainFixedEnvCfg as VIDEO_ENV_CFG
else:
    VIDEO_ENV_CFG = GuideTerrainEnvCfg


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": VIDEO_ENV_CFG, "rsl_rl_cfg_entry_point": V9_PPO_CFG},
)


# The robot's head points along base -x (see v8_env_cfg for the measurement), so
# every camera offset below is expressed in that direction, not +x.
def _set_camera(env, view: str) -> None:
    root = env.unwrapped.scene["robot"].data.root_pos_w[0]
    x, y, z = (float(v) for v in root)
    if view == "rear_quarter":
        # Behind the robot means +x, since forward is -x.
        eye = [x + 2.2, y - 1.8, z + 1.4]
        target = [x - 0.4, y, z + 0.05]
    else:
        eye = [x, y - 2.6, z + 1.0]
        target = [x - 0.2, y, z + 0.05]
    env.unwrapped.sim.set_camera_view(eye=eye, target=target)


def _create_head_marker() -> VisualizationMarkers:
    """A marker on the robot's head so walking direction is visible on video."""
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/v9_head_marker",
            markers={
                "head": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 1.0, 0.10)),
                )
            },
        )
    )


def _update_head_marker(env, marker) -> None:
    import torch as _torch
    from isaaclab.utils.math import quat_apply

    robot = env.unwrapped.scene["robot"]
    # Head offset in the base frame: forward is -x.
    offset = _torch.tensor([[-0.30, 0.0, 0.05]], device=robot.data.root_pos_w.device)
    position = robot.data.root_pos_w[:1] + quat_apply(robot.data.root_quat_w[:1], offset)
    marker.visualize(position)


def _pin_stair_level(raw_env, level: int) -> None:
    """Move the one video environment onto the requested stair row."""
    terrain = raw_env.unwrapped.scene.terrain
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types
    ]
    # ``max_init_terrain_level = level`` samples anywhere from row 0 through
    # ``level``; it does not pin a row. Reset after changing the origin so the
    # recording cannot silently land on the 0.05 m stairs it claims to exceed.
    raw_env.reset()


class TrackingWrapper(gym.Wrapper):
    def __init__(self, env, view):
        super().__init__(env)
        self.view = view
        self.marker = _create_head_marker()

    def reset(self, **kw):
        out = self.env.reset(**kw)
        _set_camera(self.env, self.view)
        _update_head_marker(self.env, self.marker)
        return out

    def step(self, action):
        out = self.env.step(action)
        _set_camera(self.env, self.view)
        _update_head_marker(self.env, self.marker)
        return out


def main() -> None:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    env_cfg = VIDEO_ENV_CFG()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env_cfg.episode_length_s = 60.0
    env_cfg.observations.policy.enable_corruption = False
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (60.0, 60.0)
    cmd.rel_standing_envs = 0.0
    # --speed is a guide-frame forward speed; the command lives in the base frame.
    from guide_frame import guide_forward_speed_range_to_base

    cmd.ranges.lin_vel_x = guide_forward_speed_range_to_base((args_cli.speed, args_cli.speed))
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.debug_vis = False
    if args_cli.scenario == "stairs_up":
        env_cfg.scene.terrain.terrain_generator = STAIRS_UP_EVAL_TERRAIN_CFG
        # A single env records one tile, so pin it to a realistic building stair
        # rather than letting it land on the 0.05 m row.
        env_cfg.scene.terrain.max_init_terrain_level = STAIRS_NUM_LEVELS - 3
        env_cfg.curriculum = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}

    agent_cfg = copy.deepcopy(V9_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    prefix = f"v9_{name}_{args_cli.scenario}_{args_cli.view}_speed{int(args_cli.speed*100):03d}"

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array")
    if args_cli.scenario == "stairs_up":
        _pin_stair_level(env, STAIRS_NUM_LEVELS - 3)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        name_prefix=prefix,
        disable_logger=True,
    )
    env = TrackingWrapper(env, args_cli.view)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        with torch.inference_mode():
            for _ in range(args_cli.video_length + 5):
                obs = env.step(policy(obs))[0]
    finally:
        env.close()
    print(f"[VIDEO] saved under {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
