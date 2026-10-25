"""Record a V8 checkpoint rollout for visual gait comparison."""

import argparse
import copy
import os

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Flat-Guide-Spot-Recipe-Video-v8"

parser = argparse.ArgumentParser(description="Record a V8 Spot-recipe rollout.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--view", choices=("side", "rear_quarter"), required=True)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--output_dir", default="output_video_v8")
# Which robot/config the checkpoint was trained with. Survival is a gate and
# the termination set decides it, so the environment must match.
parser.add_argument("--env", choices=("v8", "v10_flat", "v10_1_flat", "v10_2_flat", "v10_3_flat", "v10_4_terrain"), default="v8")
# See evaluate_v8 for how the v2 robot's fore-aft axis was measured.
parser.add_argument("--forward-axis", choices=("-x", "+y"), default="-x")
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

from v8_env_cfg import GuideSpotRecipeEnvCfg, V8_GUIDE_FORWARD_SPEED_RANGE
from v8_ppo_cfg import V8_PPO_CFG

if args_cli.env == "v10_4_terrain":
    from v10_4_terrain_env_cfg import GuideV2BlindTerrainEnvCfg as VIDEO_ENV_CFG
elif args_cli.env == "v10_3_flat":
    from v10_3_flat_env_cfg import GuideV2FlatSpawnFixedEnvCfg as VIDEO_ENV_CFG
elif args_cli.env == "v10_2_flat":
    from v10_2_flat_env_cfg import GuideV2FlatDutyLiveEnvCfg as VIDEO_ENV_CFG
elif args_cli.env == "v10_1_flat":
    from v10_1_flat_env_cfg import GuideV2FlatForwardEnvCfg as VIDEO_ENV_CFG
elif args_cli.env == "v10_flat":
    from v10_flat_env_cfg import GuideV2FlatEnvCfg as VIDEO_ENV_CFG
else:
    VIDEO_ENV_CFG = GuideSpotRecipeEnvCfg


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": VIDEO_ENV_CFG, "rsl_rl_cfg_entry_point": V8_PPO_CFG},
)


# The robot's head points along base -x (see v8_env_cfg for the measurement), so
# every camera offset below is expressed in that direction, not +x.
def _set_camera(env, view: str) -> None:
    root = env.unwrapped.scene["robot"].data.root_pos_w[0]
    x, y, z = (float(v) for v in root)
    if view == "rear_quarter":
        if args_cli.forward_axis == "+y":
            # Behind the robot means -y, since forward is +y.
            eye = [x + 1.8, y - 2.2, z + 1.4]
            target = [x, y + 0.4, z + 0.05]
        else:
            # Behind the robot means +x, since forward is -x.
            eye = [x + 2.2, y - 1.8, z + 1.4]
            target = [x - 0.4, y, z + 0.05]
    else:
        if args_cli.forward_axis == "+y":
            eye = [x - 2.6, y, z + 1.0]
            target = [x, y + 0.2, z + 0.05]
        else:
            eye = [x, y - 2.6, z + 1.0]
            target = [x - 0.2, y, z + 0.05]
    env.unwrapped.sim.set_camera_view(eye=eye, target=target)


def _create_head_marker() -> VisualizationMarkers:
    """A marker on the robot's head so walking direction is visible on video."""
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/v8_head_marker",
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
    # Head offset in the base frame.
    if args_cli.forward_axis == "+y":
        offset = _torch.tensor([[0.0, 0.45, 0.05]], device=robot.data.root_pos_w.device)
    else:
        offset = _torch.tensor([[-0.30, 0.0, 0.05]], device=robot.data.root_pos_w.device)
    position = robot.data.root_pos_w[:1] + quat_apply(robot.data.root_quat_w[:1], offset)
    marker.visualize(position)


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
    # This must branch on --forward-axis exactly as evaluate_v8.py does. It did
    # not, and the V10.1 videos were rendered with the command on base -x while
    # the camera and head marker were placed for +y -- so the robot was filmed
    # from the right side while being asked to travel in a direction its policy
    # was never trained on.
    from guide_frame import guide_forward_speed_range_to_base

    if args_cli.forward_axis == "+y":
        cmd.ranges.lin_vel_x = (0.0, 0.0)
        cmd.ranges.lin_vel_y = (args_cli.speed, args_cli.speed)
    else:
        cmd.ranges.lin_vel_x = guide_forward_speed_range_to_base(
            (args_cli.speed, args_cli.speed)
        )
        cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.debug_vis = False
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}

    agent_cfg = copy.deepcopy(V8_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
    prefix = f"v8_{name}_{args_cli.view}_speed{int(args_cli.speed*100):03d}"

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array")
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
