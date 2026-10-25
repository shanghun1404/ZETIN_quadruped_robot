import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


PROTOCOLS = {
    "pilot": {"speed": 0.15, "duration": 6.0, "num_envs": 64},
    "release": {"speed": 0.40, "duration": 12.0, "num_envs": 64},
}
EVALUATION_TASK_ID = "Isaac-Velocity-Flat-For-Reference-Eval-v0"
EXPECTED_CHECKPOINT_SHA256 = (
    "3f52fb9eb8bfdfd23294a2e24025f7ae8bb17ba9e72d8e117788d31cd62c82fc"
)
FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
WARMUP_DURATION_S = 2.0


parser = argparse.ArgumentParser(description="Evaluate the recovered July 3 Flat policy.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="flat_reference_gate.json")
parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="pilot")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from flat_reference_gate import base_x_command, evaluate_comparable_gate
from for_env_cfg import ForFlatEnvCfg
from for_ppo_cfg import ForFlatPPORunnerCfg


gym.register(
    id=EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForFlatEnvCfg,
        "rsl_rl_cfg_entry_point": ForFlatPPORunnerCfg,
    },
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_yaw_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def contact_forces(
    env, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return torch.max(torch.linalg.vector_norm(history, dim=-1), dim=1).values


def normalized_foot_impact(
    env, sensor_cfg: SceneEntityCfg, forces: torch.Tensor
) -> torch.Tensor:
    masses = env.scene["robot"].data.default_mass.to(
        device=forces.device, dtype=forces.dtype
    )
    body_weight = masses.sum(dim=1, keepdim=True) * 9.81
    nominal_force = body_weight / max(forces.shape[1], 1)
    normalized = torch.clamp(forces / nominal_force, 0.0, 4.0) / 4.0
    return torch.clamp(normalized * 4.0 - 2.0, 0.0, 1.0).square().mean(dim=1)


def record_streak(
    streak: torch.Tensor,
    maximum: torch.Tensor,
    condition: torch.Tensor,
    alive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    streak = torch.where(
        alive,
        torch.where(condition, streak + 1, torch.zeros_like(streak)),
        streak,
    )
    return streak, torch.maximum(maximum, streak)


def configure_deterministic_evaluation(env_cfg, protocol: dict[str, float]) -> None:
    env_cfg.scene.num_envs = int(protocol["num_envs"])
    env_cfg.episode_length_s = 60.0
    env_cfg.seed = 42
    env_cfg.sim.device = args_cli.device
    env_cfg.curriculum = None
    env_cfg.actions.joint_pos.scale = 0.05
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False

    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (60.0, 60.0)
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    base_speed = base_x_command(protocol["speed"])
    command.ranges.lin_vel_x = (base_speed, base_speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)

    env_cfg.events.physics_material.params["static_friction_range"] = (0.9, 0.9)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.7, 0.7)
    env_cfg.events.push_robot = None
    env_cfg.events.base_com = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    reset_pose = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.4, 0.4),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    zero_velocity = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["pose_range"] = reset_pose
    env_cfg.events.reset_base.params["velocity_range"] = zero_velocity
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def actor_input_dim(policy) -> int:
    for module in policy.actor.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    raise RuntimeError("actor contains no Linear input layer")


def main() -> bool:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    checkpoint_sha256 = sha256_file(args_cli.checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )

    protocol = PROTOCOLS[args_cli.protocol]
    num_envs = int(protocol["num_envs"])
    env_cfg = ForFlatEnvCfg()
    configure_deterministic_evaluation(env_cfg, protocol)
    agent_cfg = ForFlatPPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device
    gym_env = gym.make(EVALUATION_TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint, load_optimizer=False)
        actual_actor_input_dim = actor_input_dim(runner.alg.policy)
        if actual_actor_input_dim != 48:
            raise RuntimeError(
                f"expected a 48-observation actor, got {actual_actor_input_dim}"
            )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs_result = env.get_observations()
        obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        foot_cfg = SceneEntityCfg(
            "contact_forces", body_names=FOOT_NAMES, preserve_order=True
        )
        foot_cfg.resolve(base_env.scene)
        warmup_alive = torch.ones(
            num_envs, dtype=torch.bool, device=base_env.device
        )
        warmup_termination_counts = {
            name: 0 for name in base_env.termination_manager.active_terms
        }
        warmup_steps = round(WARMUP_DURATION_S / base_env.step_dt)
        for _ in range(warmup_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            for name in warmup_termination_counts:
                term_dones = base_env.termination_manager.get_term(name).bool()
                warmup_termination_counts[name] += (
                    warmup_alive & term_dones
                ).sum().item()
            warmup_alive &= ~dones.bool()

        steps = round(float(protocol["duration"]) / base_env.step_dt)
        start_y = robot.data.root_pos_w[:, 1].clone()
        max_lateral = torch.zeros(num_envs, device=base_env.device)
        max_heading = torch.zeros(num_envs, device=base_env.device)
        height_min = robot.data.root_pos_w[:, 2].clone()
        height_max = height_min.clone()
        wrong_streak = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
        max_wrong_streak = torch.zeros_like(wrong_streak)
        under_support_streak = torch.zeros_like(wrong_streak)
        max_under_support_streak = torch.zeros_like(wrong_streak)
        flight_streak = torch.zeros_like(wrong_streak)
        max_flight_streak = torch.zeros_like(wrong_streak)
        alive = warmup_alive.clone()
        termination_counts = {
            name: 0 for name in base_env.termination_manager.active_terms
        }
        sum_forward = 0.0
        sum_height = 0.0
        sum_yaw_sq = 0.0
        sum_vz_sq = 0.0
        sum_roll_pitch_sq = 0.0
        sum_impact = 0.0
        support_good = 0
        flight_samples = 0
        total_samples = 0
        all_terms_finite = True

        for _ in range(steps):
            root_lin_b = robot.data.root_lin_vel_b
            root_ang_b = robot.data.root_ang_vel_b
            semantic_forward = -root_lin_b[:, 0]
            heading = torch.abs(quaternion_yaw_wxyz(robot.data.root_quat_w))
            forces = contact_forces(base_env, foot_cfg)
            contact_count = (forces > 5.0).float().sum(dim=-1)
            height = robot.data.root_pos_w[:, 2]
            impact = normalized_foot_impact(base_env, foot_cfg, forces)
            wrong_streak, max_wrong_streak = record_streak(
                wrong_streak,
                max_wrong_streak,
                semantic_forward < 0.0,
                alive,
            )
            under_support_streak, max_under_support_streak = record_streak(
                under_support_streak,
                max_under_support_streak,
                contact_count < 2.0,
                alive,
            )
            flight_streak, max_flight_streak = record_streak(
                flight_streak,
                max_flight_streak,
                contact_count == 0.0,
                alive,
            )
            lateral = torch.abs(robot.data.root_pos_w[:, 1] - start_y)
            max_lateral = torch.where(
                alive, torch.maximum(max_lateral, lateral), max_lateral
            )
            max_heading = torch.where(
                alive, torch.maximum(max_heading, heading), max_heading
            )
            height_min = torch.where(alive, torch.minimum(height_min, height), height_min)
            height_max = torch.where(alive, torch.maximum(height_max, height), height_max)
            sum_forward += semantic_forward[alive].sum().item()
            sum_height += height[alive].sum().item()
            sum_yaw_sq += root_ang_b[alive, 2].square().sum().item()
            sum_vz_sq += root_lin_b[alive, 2].square().sum().item()
            sum_roll_pitch_sq += root_ang_b[alive, :2].square().sum().item()
            sum_impact += impact[alive].sum().item()
            support_good += (contact_count[alive] >= 2.0).sum().item()
            flight_samples += (contact_count[alive] == 0.0).sum().item()
            total_samples += alive.sum().item()

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            for name in termination_counts:
                term_dones = base_env.termination_manager.get_term(name).bool()
                termination_counts[name] += (alive & term_dones).sum().item()
            all_terms_finite &= bool(
                torch.isfinite(base_env.reward_manager._step_reward[alive]).all().item()
            )
            alive &= ~dones.bool()

        if total_samples == 0:
            raise RuntimeError("evaluation collected no alive samples")
        metrics = {
            "mean_forward_speed_m_s": sum_forward / total_samples,
            "max_wrong_direction_s": max_wrong_streak.max().item() * base_env.step_dt,
            "p95_lateral_displacement_m": torch.quantile(max_lateral, 0.95).item(),
            "p95_heading_drift_deg": math.degrees(
                torch.quantile(max_heading, 0.95).item()
            ),
            "rms_yaw_rate_rad_s": math.sqrt(sum_yaw_sq / total_samples),
            "support_ge2_fraction": support_good / total_samples,
            "max_under_support_s": max_under_support_streak.max().item()
            * base_env.step_dt,
            "flight_fraction": flight_samples / total_samples,
            "max_flight_s": max_flight_streak.max().item() * base_env.step_dt,
            "mean_base_height_m": sum_height / total_samples,
            "p95_height_ptp_m": torch.quantile(height_max - height_min, 0.95).item(),
            "rms_vertical_velocity_m_s": math.sqrt(sum_vz_sq / total_samples),
            "rms_roll_pitch_rate_rad_s": math.sqrt(
                sum_roll_pitch_sq / total_samples
            ),
            "mean_normalized_foot_impact": sum_impact / total_samples,
            "survival_fraction": alive.float().mean().item(),
            "warmup_survival_fraction": warmup_alive.float().mean().item(),
            "warmup_termination_counts": warmup_termination_counts,
            "termination_counts": termination_counts,
            "all_terms_finite": all_terms_finite,
        }
        gates = evaluate_comparable_gate(metrics, args_cli.protocol)
        payload = {
            "provenance": {
                "checkpoint": os.path.abspath(args_cli.checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "actor_observation_dim": 48,
                "policy": "ActorCritic MLP 128x128x128",
                "action_scale": 0.05,
                "semantic_forward_axis": "-base_x",
                "reward_inventory_comparable": False,
            },
            "protocol": {
                "name": args_cli.protocol,
                "semantic_speed_m_s": protocol["speed"],
                "base_x_command_m_s": base_x_command(protocol["speed"]),
                "duration_s": protocol["duration"],
                "warmup_duration_s": WARMUP_DURATION_S,
                "reset_height_offset_m": 0.4,
                "num_envs": num_envs,
                "step_dt_s": base_env.step_dt,
            },
            "metrics": metrics,
            "gates": gates,
            "passed": all(gates.values()),
        }
        Path(args_cli.output).write_text(json.dumps(payload, indent=2) + "\n")
        for name, passed in gates.items():
            print(f"[GATE] {name}={'PASS' if passed else 'FAIL'}")
        return payload["passed"]
    finally:
        env.close()


if __name__ == "__main__":
    passed = main()
    simulation_app.close()
    raise SystemExit(0 if passed else 1)
