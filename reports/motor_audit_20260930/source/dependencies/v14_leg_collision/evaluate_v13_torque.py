"""Compare V12 and V13 depth-driven walking with physics-step torque metrics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Guide-RobotV2-v13-torque-eval"
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("HAA", "HFE", "KFE"))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=("v12", "v13"), required=True)
parser.add_argument("--actor-checkpoint", required=True)
parser.add_argument("--student-checkpoint", required=True)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--trace-output", help="Optional compressed physics-step torque trace (.npz)")
parser.add_argument("--protocol", choices=("legacy", "fixed_v2"), default="legacy")
parser.add_argument("--heading-control", action="store_true")
parser.add_argument("--residual-checkpoint", help="Optional V14 bounded residual; actor-checkpoint remains the frozen base")
parser.add_argument("--trajectory-output", help="Optional path/pose/command/terrain trace (.npz)")
parser.add_argument("--video-dir", help="Record selected environments into a new directory")
parser.add_argument("--video-env-ids", type=int, nargs="+", default=[])
parser.add_argument("--video-view", choices=("side", "rear_quarter"), default="rear_quarter")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--warmup-seconds", type=float, default=2.0)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--forward-speed", type=float, default=0.4)
parser.add_argument("--effort-guard-fraction", type=float, help="Fraction of the V13 motor torque-speed envelope")
parser.add_argument("--scenario", choices=("flat", "rough", "stairs_up", "stairs_down"), default="flat")
parser.add_argument("--terrain-level", type=int)
parser.add_argument("--enable-self-collisions", action="store_true", help="Use the derived V14 USD inter-leg colliders")
parser.add_argument("--measure-leg-clearance", action="store_true", help="Measure V14 capsule gaps at every control step")
parser.add_argument("--leg-capsule-radius-m", type=float, default=0.02)
parser.add_argument("--filter-leg-terrain", action="store_true", help="Filter V14 capsules against terrain before physics starts")
parser.add_argument("--control", choices=("isaaclab", "ros2"), default="isaaclab",
                    help="ros2: policy, LPF and torque guard run as ROS2 nodes; PhysX joint drives do PD")
parser.add_argument("--ros2-namespace", default="/v14")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.trajectory_output and args_cli.protocol != "fixed_v2":
    parser.error("--trajectory-output requires --protocol fixed_v2 with known terrain geometry")
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from rsl_rl.runners import OnPolicyRunner

from v11_terrain_env_cfg import GuideV2ScannerTerrainEnvCfg
from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG
from v12_camera import make_v12_depth_camera_cfg
from v13_depth_policy import make_v13_depth_wrapper
from v13_torque_env_cfg import GuideV13TorqueEnvCfg
from v13_torque_metrics import summarize_actuation
from v13_torque_ppo_cfg import V13_TORQUE_PPO_CFG
from v14_clearance import interleg_clearances_m, summarize_interleg_clearance
from v14_terrain_filter import apply_v14_ground_filter
from v9_env_cfg import (
    STAIRS_DOWN_EVAL_TERRAIN_CFG,
    STAIRS_NUM_LEVELS,
    STAIRS_STEP_HEIGHT_RANGE,
    STAIRS_UP_EVAL_TERRAIN_CFG,
)
from v11_video_geometry import stair_step_height_m
from v12_eval_metrics import stair_traversal_mask


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path in (args_cli.actor_checkpoint, args_cli.student_checkpoint, args_cli.usd_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if args_cli.num_envs <= 0 or args_cli.seconds <= 0 or args_cli.warmup_seconds < 0:
        raise ValueError("num-envs, seconds and warmup-seconds must be valid")
    if args_cli.effort_guard_fraction is not None:
        if args_cli.variant != "v13" or not 0.0 < args_cli.effort_guard_fraction <= 1.0:
            raise ValueError("effort guard requires v13 and a fraction in (0, 1]")
    if args_cli.measure_leg_clearance and not 0.01 <= args_cli.leg_capsule_radius_m <= 0.025:
        raise ValueError("measured leg capsule radius must be in [0.01, 0.025] m")
    if args_cli.filter_leg_terrain and not args_cli.enable_self_collisions:
        raise ValueError("terrain filtering requires V14 self collisions")
    if args_cli.heading_control and args_cli.protocol != "fixed_v2":
        raise ValueError("heading control requires the fixed_v2 protocol")
    if args_cli.control == "ros2" and (
        args_cli.variant != "v13" or args_cli.num_envs != 1 or args_cli.effort_guard_fraction is not None
        or args_cli.residual_checkpoint or args_cli.heading_control
    ):
        raise ValueError("ros2 control drives one v13-variant robot; the guard is a ROS2 node parameter")
    residual_manifest = None
    if args_cli.residual_checkpoint:
        from v14_residual_policy import load_residual_manifest

        if (args_cli.variant != "v13" or args_cli.protocol != "fixed_v2"
                or args_cli.effort_guard_fraction != .99 or not args_cli.enable_self_collisions
                or not args_cli.filter_leg_terrain or args_cli.leg_capsule_radius_m != .02):
            raise ValueError("residual evaluation requires the reviewed V14 fixed protocol and guard")
        residual_manifest = load_residual_manifest(
            args_cli.residual_checkpoint, base_checkpoint=args_cli.actor_checkpoint,
            student_checkpoint=args_cli.student_checkpoint, usd_path=args_cli.usd_path,
            heading_control=args_cli.heading_control,
        )
    if bool(args_cli.video_dir) != bool(args_cli.video_env_ids):
        raise ValueError("video-dir and video-env-ids must be supplied together")
    if len(set(args_cli.video_env_ids)) != len(args_cli.video_env_ids) or any(
        not 0 <= env_id < args_cli.num_envs for env_id in args_cli.video_env_ids
    ):
        raise ValueError("video environment IDs must be unique and in range")

    env_type = GuideV13TorqueEnvCfg if args_cli.variant == "v13" else GuideV2ScannerTerrainEnvCfg
    env_cfg = env_type()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.protocol == "fixed_v2":
        env_cfg.scene.env_spacing = 12.0
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
    if args_cli.enable_self_collisions:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    if args_cli.filter_leg_terrain:
        env_cfg.scene.replicate_physics = False
        env_cfg.events.v14_ground_filter = EventTerm(
            func=apply_v14_ground_filter, mode="prestartup"
        )
    if args_cli.effort_guard_fraction is not None:
        env_cfg.actions.joint_pos.effort_guard_fraction = args_cli.effort_guard_fraction
    if args_cli.control == "ros2":
        from v14_ros2_bridge import V14Ros2JointTargetActionCfg, implicit_joint_drives

        env_cfg.actions.joint_pos = V14Ros2JointTargetActionCfg(
            asset_name="robot", namespace=args_cli.ros2_namespace
        )
        implicit_joint_drives(env_cfg.scene.robot)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.seconds + args_cli.warmup_seconds + 10.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.curriculum = None
    env_cfg.events.push_robot = None
    terrain_level = None
    geometry = {}
    if args_cli.scenario == "flat":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    else:
        terrain_level = args_cli.terrain_level
        if terrain_level is None:
            terrain_level = 4 if args_cli.scenario == "rough" else 5
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        if args_cli.scenario == "stairs_up":
            env_cfg.scene.terrain.terrain_generator = copy.deepcopy(STAIRS_UP_EVAL_TERRAIN_CFG)
        elif args_cli.scenario == "stairs_down":
            env_cfg.scene.terrain.terrain_generator = copy.deepcopy(STAIRS_DOWN_EVAL_TERRAIN_CFG)
        generator = env_cfg.scene.terrain.terrain_generator
        if generator is None or not 0 <= terrain_level < int(generator.num_rows):
            raise ValueError("terrain level must exist in the selected generator")
        generator.curriculum = False
        env_cfg.scene.terrain.max_init_terrain_level = terrain_level
        if args_cli.protocol == "fixed_v2":
            from v14_eval_protocol import fixed_eval_generator

            generator, geometry = fixed_eval_generator(
                generator, args_cli.scenario, terrain_level, args_cli.num_envs
            )
            env_cfg.scene.terrain.terrain_generator = generator
            env_cfg.scene.terrain.max_init_terrain_level = 0
    if args_cli.variant == "v12":
        env_cfg.scene.depth_camera = make_v12_depth_camera_cfg(env_cfg.decimation * env_cfg.sim.dt)
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (1.0e6, 1.0e6)
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (args_cli.forward_speed, args_cli.forward_speed)
    command.ranges.ang_vel_z = (0.0, 0.0)
    if args_cli.protocol == "fixed_v2":
        command.debug_vis = False
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }

    if args_cli.heading_control:
        from v14_heading_command import make_lane_command_cfg
        env_cfg.commands.base_velocity = make_lane_command_cfg(command)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)

    agent_cfg = copy.deepcopy(V13_TORQUE_PPO_CFG if args_cli.variant == "v13" else V11_TERRAIN_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device
    gym.register(id=TASK_ID, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    if terrain_level is not None:
        base = raw_env.unwrapped
        terrain = base.scene.terrain
        terrain.terrain_levels.fill_(0 if args_cli.protocol == "fixed_v2" else terrain_level)
        env_ids = torch.arange(base.num_envs, device=base.device)
        terrain.terrain_types[:] = env_ids % int(generator.num_cols)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        if args_cli.protocol == "fixed_v2":
            origins = terrain.env_origins.detach().cpu().numpy()
            geometry["unique_origins"] = len(np.unique(origins, axis=0))
            geometry["origin_z_m"] = origins[:, 2].tolist()
            if geometry["unique_origins"] != args_cli.num_envs:
                raise RuntimeError("fixed protocol environments overlap")
            if "full_rise_m" in geometry:
                expected_z = geometry["full_rise_m"] * (-1 if args_cli.scenario == "stairs_up" else 1)
                if not np.allclose(origins[:, 2], expected_z, atol=1e-4):
                    raise RuntimeError("actual stair origins disagree with fixed geometry")
        raw_env.reset()
    DepthWrapper = make_v13_depth_wrapper()
    env = DepthWrapper(raw_env, student_checkpoint=args_cli.student_checkpoint, clip_actions=agent_cfg.clip_actions)
    video = None
    try:
        # With --control ros2 this in-process policy only shadows the ROS2 nodes for the parity report.
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.actor_checkpoint, load_optimizer=False, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        if residual_manifest is not None:
            from v14_residual_policy import ResidualPolicyWrapper, module_digest, residual_agent_cfg

            env = ResidualPolicyWrapper(env, runner.alg.policy, max_rad=residual_manifest["max_rad"])
            frozen_before = (module_digest(env.frozen_actor), module_digest(env.env._student))
            residual_runner = OnPolicyRunner(env, residual_agent_cfg(agent_cfg).to_dict(), log_dir=None, device=agent_cfg.device)
            residual_runner.load(args_cli.residual_checkpoint, load_optimizer=False, map_location=agent_cfg.device)
            policy = residual_runner.get_inference_policy(device=agent_cfg.device)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        lookup = {name: index for index, name in enumerate(robot.joint_names)}
        indices = torch.tensor([lookup[name] for name in JOINT_NAMES], device=base_env.device)
        if args_cli.measure_leg_clearance:
            profile_names = tuple(
                f"{leg}_{part}_profile"
                for leg in ("FL", "FR", "RL", "RR")
                for part in ("thigh", "calf")
            )
            body_lookup = {name: index for index, name in enumerate(robot.body_names)}
            profile_indices = torch.tensor(
                [body_lookup[name] for name in profile_names], device=base_env.device
            )
            profile_positions = []
            profile_quaternions = []
            profile_alive = []
        joint_limits = torch.tensor([24.8 if name.endswith("HAA") else 48.0 for name in JOINT_NAMES])
        physics_dt = float(env_cfg.sim.dt)
        control_dt = float(base_env.step_dt)
        warmup_steps = round(args_cli.warmup_seconds / control_dt)
        measured_steps = round(args_cli.seconds / control_dt)
        if measured_steps < 1:
            raise ValueError("measurement interval rounds to zero steps")
        if args_cli.video_dir:
            from v14_video import RolloutVideos

            video = RolloutVideos(
                base_env.sim, robot, args_cli.video_dir, args_cli.video_env_ids,
                view=args_cli.video_view, fps=1.0 / (2 * control_dt),
            )

        applied = []
        requested = []
        velocity = []
        alive_samples = []
        recording = False
        alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device=base_env.device)
        first_termination_s = [None] * args_cli.num_envs
        first_termination_terms = [[] for _ in range(args_cli.num_envs)]
        original_write = base_env.scene.write_data_to_sim

        def write_and_record() -> None:
            original_write()
            if recording:
                applied.append(robot.data.applied_torque[:, indices].detach().cpu().clone())
                requested.append(robot.data.computed_torque[:, indices].detach().cpu().clone())
                velocity.append(robot.data.joint_vel[:, indices].detach().cpu().clone())
                alive_samples.append(alive.cpu().clone())

        base_env.scene.write_data_to_sim = write_and_record
        observations = env.get_observations()
        initial_xy = None
        last_xy = None
        forward_xy = None
        initial_height = None
        max_progress = torch.zeros(args_cli.num_envs, device=base_env.device)
        vertical_after_approach = torch.zeros(args_cli.num_envs, device=base_env.device)
        speed_sum = 0.0
        speed_count = 0
        trajectory = {key: [] for key in ("positions", "quaternions", "ground_height", "valid",
                                         "linear_velocity_w", "angular_velocity_b", "commands")}
        record_path = args_cli.protocol == "fixed_v2" or bool(args_cli.trajectory_output)

        with torch.inference_mode():
            for step in range(warmup_steps + measured_steps):
                if record_path and step >= warmup_steps:
                    if args_cli.scenario == "flat":
                        ground = torch.zeros(args_cli.num_envs, device=base_env.device)
                    else:
                        from v14_eval_protocol import ground_below_robot
                        ground = ground_below_robot(base_env)
                    values = dict(positions=robot.data.root_pos_w, quaternions=robot.data.root_quat_w,
                                  ground_height=ground, valid=alive,
                                  linear_velocity_w=robot.data.root_lin_vel_w,
                                  angular_velocity_b=robot.data.root_ang_vel_b,
                                  commands=base_env.command_manager.get_command("base_velocity"))
                    for key, value in values.items():
                        trajectory[key].append(value.detach().cpu().numpy().copy())
                if step == warmup_steps:
                    initial_xy = robot.data.root_pos_w[:, :2].clone()
                    last_xy = initial_xy.clone()
                    initial_height = robot.data.root_pos_w[:, 2].clone()
                    q = robot.data.root_quat_w
                    forward_xy = torch.stack(
                        (2.0 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3]),
                         1.0 - 2.0 * (q[:, 1].square() + q[:, 3].square())),
                        dim=-1,
                    )
                if step >= warmup_steps and bool(alive.any()):
                    speed_sum += float(robot.data.root_lin_vel_b[alive, 1].sum())
                    speed_count += int(alive.sum())
                    last_xy[alive] = robot.data.root_pos_w[alive, :2]
                    if args_cli.scenario in ("stairs_up", "stairs_down"):
                        progress = torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - initial_xy, dim=-1)
                        max_progress = torch.where(alive, torch.maximum(max_progress, progress), max_progress)
                        reached = alive & (progress >= 2.1)
                        height_change = robot.data.root_pos_w[:, 2] - initial_height
                        if args_cli.scenario == "stairs_down":
                            height_change = -height_change
                        vertical_after_approach = torch.where(
                            reached,
                            torch.maximum(vertical_after_approach, height_change),
                            vertical_after_approach,
                        )
                if video is not None and step >= warmup_steps and (step - warmup_steps) % 2 == 0:
                    # Capture before stepping: the environment auto-resets terminated robots.
                    video.capture(alive)
                recording = step >= warmup_steps
                observations, _, dones, _ = env.step(policy(observations))
                recording = False
                if step >= warmup_steps:
                    for env_id in torch.nonzero(alive & dones.bool()).flatten().tolist():
                        first_termination_s[env_id] = (step - warmup_steps + 1) * control_dt
                        first_termination_terms[env_id] = [
                            name for name in base_env.termination_manager.active_terms
                            if bool(base_env.termination_manager.get_term(name)[env_id])
                        ]
                    alive &= ~dones.bool()
                    if args_cli.measure_leg_clearance:
                        profile_positions.append(
                            robot.data.body_pos_w[:, profile_indices].detach().cpu().numpy().copy()
                        )
                        profile_quaternions.append(
                            robot.data.body_quat_w[:, profile_indices].detach().cpu().numpy().copy()
                        )
                        profile_alive.append(alive.cpu().numpy().copy())

        base_env.scene.write_data_to_sim = original_write
        expected = measured_steps * env_cfg.decimation
        if len(applied) != expected:
            raise RuntimeError(f"recorded {len(applied)} physics steps, expected {expected}")
        applied_tensor = torch.stack(applied)
        requested_tensor = torch.stack(requested)
        velocity_tensor = torch.stack(velocity)
        alive_tensor = torch.stack(alive_samples)
        report = {
            "variant": args_cli.variant,
            "joint_order": JOINT_NAMES,
            "actuation": summarize_actuation(
                applied_tensor, requested_tensor, velocity_tensor,
                alive_tensor, joint_limits, physics_dt_s=physics_dt,
            ),
            "mean_forward_speed_m_s": speed_sum / speed_count if speed_count else 0.0,
            "mean_directed_progress_m": float(((last_xy - initial_xy) * forward_xy).sum(dim=-1).mean()),
            "survived_envs": int(alive.sum()),
            "initial_envs": args_cli.num_envs,
            "metadata": {
                "actor_checkpoint": args_cli.actor_checkpoint,
                "actor_sha256": sha256(args_cli.actor_checkpoint),
                "student_checkpoint": args_cli.student_checkpoint,
                "student_sha256": sha256(args_cli.student_checkpoint),
                "usd_sha256": sha256(args_cli.usd_path),
                "seed": args_cli.seed,
                "warmup_s": warmup_steps * control_dt,
                "measured_s": measured_steps * control_dt,
                "control_dt_s": control_dt,
                "physics_dt_s": physics_dt,
                "command_forward_m_s": args_cli.forward_speed,
                "effort_guard_fraction": args_cli.effort_guard_fraction,
                "control": args_cli.control,
                "ros2_namespace": args_cli.ros2_namespace if args_cli.control == "ros2" else None,
                "scenario": args_cli.scenario,
                "protocol": args_cli.protocol,
                "heading_control": args_cli.heading_control,
                "geometry": geometry,
                "terrain_level": terrain_level,
                "self_collisions_enabled": args_cli.enable_self_collisions,
                "measure_leg_clearance": args_cli.measure_leg_clearance,
                "leg_capsule_radius_m": args_cli.leg_capsule_radius_m if args_cli.measure_leg_clearance else None,
                "leg_terrain_filtered": args_cli.filter_leg_terrain,
            },
        }
        if args_cli.control == "ros2":
            report["metadata"]["ros2_parity"] = base_env.action_manager.get_term("joint_pos").parity_summary()
        if residual_manifest is not None:
            unchanged = frozen_before == (module_digest(env.frozen_actor), module_digest(env.env._student))
            if not unchanged or env.max_observed_delta_rad > residual_manifest["max_rad"] + 1e-6:
                raise RuntimeError("residual evaluation violated frozen/limit contract")
            report["metadata"]["residual"] = {
                "checkpoint": args_cli.residual_checkpoint,
                "sha256": sha256(args_cli.residual_checkpoint),
                "max_rad": residual_manifest["max_rad"],
                "max_observed_delta_rad": env.max_observed_delta_rad,
                "frozen_base_student_unchanged": unchanged,
            }
        if args_cli.measure_leg_clearance:
            positions = np.stack(profile_positions)
            quaternions = np.stack(profile_quaternions)
            gaps = interleg_clearances_m(
                positions.reshape(-1, 8, 3),
                quaternions.reshape(-1, 8, 4),
                profile_names,
                radius_m=args_cli.leg_capsule_radius_m,
            ).reshape(measured_steps, args_cli.num_envs, 24)
            report["interleg_clearance"] = summarize_interleg_clearance(
                gaps, np.stack(profile_alive)
            )
        traversed = None
        if args_cli.scenario in ("stairs_up", "stairs_down"):
            step_height = stair_step_height_m(STAIRS_STEP_HEIGHT_RANGE, STAIRS_NUM_LEVELS, terrain_level)
            traversed = stair_traversal_mask(
                args_cli.scenario, alive, max_progress, vertical_after_approach,
                approach_distance_m=2.1, step_height_m=step_height,
            )
            report["stairs"] = {
                "step_height_m": step_height,
                "traversed_count": int(traversed.sum()),
                "traversal_rate": float(traversed.float().mean()),
                "mean_vertical_after_approach_m": float(vertical_after_approach.mean()),
            }
        directed_progress = ((last_xy - initial_xy) * forward_xy).sum(dim=-1)
        report["per_environment"] = [
            {
                "env_id": env_id,
                "survived": bool(alive[env_id]),
                "first_termination_s": first_termination_s[env_id],
                "first_termination_terms": first_termination_terms[env_id],
                "directed_progress_m": float(directed_progress[env_id]),
                "max_radial_progress_m": float(max_progress[env_id]),
                "vertical_after_approach_m": float(vertical_after_approach[env_id]),
                "traversed": bool(traversed[env_id]) if traversed is not None else None,
            }
            for env_id in range(args_cli.num_envs)
        ]
        if record_path:
            from v14_path_metrics import summarize_path

            arrays = {key: np.stack(value) for key, value in trajectory.items()}
            report["path"] = summarize_path(
                **{key: arrays[key] for key in ("positions", "quaternions", "ground_height", "valid")},
                alive_at_end=alive.cpu().numpy(), scenario=args_cli.scenario,
                step_height_m=geometry.get("step_height_m", 0.0),
                full_rise_m=geometry.get("full_rise_m", 0.0),
            )
            if args_cli.trajectory_output:
                path = Path(args_cli.trajectory_output)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays, alive_at_end=alive.cpu().numpy(),
                                    time_s=np.arange(measured_steps)*control_dt)
                report["trajectory_path"] = str(path)
        if video is not None:
            video.close()
            report["videos"] = video.metadata()
        output = Path(args_cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if args_cli.trace_output:
            trace_path = Path(args_cli.trace_output)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trace_path,
                applied_torque_nm=applied_tensor.numpy(),
                computed_torque_nm=requested_tensor.numpy(),
                joint_velocity_rad_s=velocity_tensor.numpy(),
                alive=alive_tensor.numpy(),
                time_s=np.arange(expected, dtype=np.float64) * physics_dt,
                joint_names=np.asarray(JOINT_NAMES),
                seed=np.asarray(args_cli.seed),
                variant=np.asarray(args_cli.variant),
            )
            report["trace_path"] = str(trace_path)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        try:
            if video is not None:
                video.close()
        finally:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # Kit's fast shutdown otherwise replaces Python's nonzero exit with 0.
        import omni.kit.app

        omni.kit.app.get_app().post_uncancellable_quit(1)
        simulation_app.close()
        raise SystemExit(1)
    simulation_app.close()
