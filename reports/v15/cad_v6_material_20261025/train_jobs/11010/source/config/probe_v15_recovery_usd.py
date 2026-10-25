"""Drop the robot in fixed poses on the V15-R plant and record where the trunk comes to rest.

Zero policy action (targets hold the stand pose through the V15 LPF/guard). A USD
without trunk collision lets fallen poses sink below the ground; the V15 recovery
USD must hold them on the trunk while a standing robot keeps its trunk off the ground.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--copies", type=int, default=4)
parser.add_argument("--seconds", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

from v15_math import STAND_ROOT_HEIGHT_M, tilt_rad  # noqa: E402
from v15_recovery_env_cfg import DROP_ROOT_HEIGHT_M, GuideV15RecoveryEnvCfg, root_height  # noqa: E402

# (name, rotation axis, angle): forward is base +y, so rolling onto a side turns about y.
POSES = (("standing", (0, 0, 1), 0.0), ("upside_down", (0, 1, 0), math.pi),
         ("left_side", (0, 1, 0), math.pi / 2), ("right_side", (0, 1, 0), -math.pi / 2),
         ("nose_down", (1, 0, 0), math.pi / 2), ("tail_down", (1, 0, 0), -math.pi / 2))


def reset_named_poses(env, env_ids) -> None:
    robot = env.scene["robot"]
    quat, height = [], []
    for env_id in env_ids.tolist():
        _, axis, angle = POSES[env_id % len(POSES)]
        quat.append([math.cos(angle / 2)] + [math.sin(angle / 2) * a for a in axis])
        height.append(STAND_ROOT_HEIGHT_M + 0.02 if angle == 0.0 else DROP_ROOT_HEIGHT_M)
    position = env.scene.env_origins[env_ids].clone()
    position[:, 2] += torch.tensor(height, device=env.device)
    pose = torch.cat((position, torch.tensor(quat, device=env.device)), dim=-1)
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    default = robot.data.default_joint_pos[env_ids]
    robot.write_joint_state_to_sim(default, torch.zeros_like(default), env_ids=env_ids)


def main() -> None:
    env_cfg = GuideV15RecoveryEnvCfg()
    env_cfg.scene.num_envs = len(POSES) * args_cli.copies
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
    env_cfg.scene.trunk_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base/base", history_length=1)
    env_cfg.events.reset_robot = EventTerm(func=reset_named_poses, mode="reset")
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.actuator_gains = None
    env_cfg.actions.joint_pos.max_delay_steps = 0
    env_cfg.episode_length_s = args_cli.seconds + 5.0
    env_cfg.observations.policy.enable_corruption = False
    gym.register(id="Isaac-Recovery-Guide-RobotV2-v15-probe", entry_point="isaaclab.envs:ManagerBasedRLEnv",
                 disable_env_checker=True)
    env = gym.make("Isaac-Recovery-Guide-RobotV2-v15-probe", cfg=env_cfg)
    try:
        base = env.unwrapped
        env.reset()
        robot, contact = base.scene["robot"], base.scene["trunk_contact"]
        zeros = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
        lowest = torch.full((base.num_envs,), math.inf, device=base.device)
        peak_trunk_force = torch.zeros(base.num_envs, device=base.device)
        steps = round(args_cli.seconds / base.step_dt)
        with torch.inference_mode():
            for _ in range(steps):
                base.step(zeros)
                lowest = torch.minimum(lowest, root_height(base))
                force = contact.data.net_forces_w[:, 0].norm(dim=-1)
                peak_trunk_force = torch.maximum(peak_trunk_force, force)
            final_force = contact.data.net_forces_w[:, 0].norm(dim=-1)
        weight = float(robot.data.default_mass.sum(-1).mean()) * 9.81
        rows = []
        for index, (name, _, _) in enumerate(POSES):
            ids = torch.arange(index, base.num_envs, len(POSES), device=base.device)
            rows.append({
                "pose": name,
                "final_root_height_m": root_height(base)[ids].tolist(),
                "lowest_root_height_m": lowest[ids].tolist(),
                "final_tilt_deg": tilt_rad(robot.data.projected_gravity_b[ids]).rad2deg().tolist(),
                "final_trunk_contact_force_body_weights": (final_force[ids] / weight).tolist(),
                "peak_trunk_contact_force_body_weights": (peak_trunk_force[ids] / weight).tolist(),
            })
        report = {"usd_path": args_cli.usd_path, "seconds": steps * base.step_dt, "robot_weight_n": weight,
                  "poses": rows}
        Path(args_cli.output).write_text(json.dumps(report, indent=2) + "\n")
        for row in rows:
            print(row["pose"], "final z", [round(v, 3) for v in row["final_root_height_m"]],
                  "lowest z", [round(v, 3) for v in row["lowest_root_height_m"]],
                  "trunk force/BW", [round(v, 2) for v in row["final_trunk_contact_force_body_weights"]], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        import omni.kit.app

        omni.kit.app.get_app().post_uncancellable_quit(1)
        simulation_app.close()
        raise SystemExit(1)
    simulation_app.close()
