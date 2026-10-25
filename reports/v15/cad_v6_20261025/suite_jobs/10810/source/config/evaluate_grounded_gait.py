import argparse
import copy
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


PROTOCOLS = {
    "pilot": {"speed": 0.15, "duration": 6.0, "num_envs": 64},
    "release": {"speed": 0.40, "duration": 12.0, "num_envs": 64},
}
EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Phase-Free-GRU-Eval-v5-2"
V5_3_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Reference-GRU-Eval-v5-3"
V5_4_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Pattern-Neutral-GRU-Eval-v5-4"
V5_6_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Phase-Diverse-GRU-Eval-v5-6"
V5_7_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Usage-Floor-GRU-Eval-v5-7"
V5_8_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Usage-Gated-GRU-Eval-v5-8"
V5_9_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Symmetric-Gate-GRU-Eval-v5-9"
V5_11_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Reference-Reset-GRU-Eval-v5-11"
V5_12_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Anti-Cheat-GRU-Eval-v5-12"
V5_13_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Measured-Anti-Cheat-GRU-Eval-v5-13"
V5_14_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Posture-Fix-GRU-Eval-v5-14"
V6_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Upright-Anti-Cheat-GRU-Eval-v6"
V7_EVALUATION_TASK_ID = "Isaac-Velocity-Flat-Guide-Small-Action-GRU-Eval-v7"


parser = argparse.ArgumentParser(description="Evaluate a fixed V5 phase-free locomotion protocol.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="phase_free_gate.json")
parser.add_argument("--protocol", choices=("pilot", "release"), default="release")
parser.add_argument(
    "--speed",
    type=float,
    default=None,
    help="Override the protocol semantic evaluation speed (m/s).",
)
parser.add_argument(
    "--variant",
    choices=("v5_2", "v5_3", "v5_4", "v5_6", "v5_7", "v5_8", "v5_9", "v5_11", "v5_12", "v5_13", "v6", "v7"),
    default="v5_2",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import rsl_rl.runners.on_policy_runner as runner_module
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from evaluation_cfg import configure_deterministic_evaluation
from grounded_gate import advance_alive_mask, evaluate_gate, evaluate_pilot_gate, record_contact_streak
from smooth_actor_critic import SmoothActorCritic
from smooth_env_cfg import GroundedGuideEnvCfg
from smooth_math import quaternion_yaw_wxyz
from smooth_ppo_cfg import SMOOTH_GUIDE_PPO_CFG
from v5_3_env_cfg import ReferenceTrainingGuideEnvCfg, V5_3_REWARD_WEIGHTS
from v5_3_ppo_cfg import V5_3_PPO_CFG
from v5_4_env_cfg import PatternNeutralGuideEnvCfg, V5_4_REWARD_WEIGHTS
from v5_4_ppo_cfg import V5_4_PPO_CFG
from v5_6_env_cfg import PhaseDiverseGuideEnvCfg, V5_6_REWARD_WEIGHTS
from v5_6_math import (
    evaluate_phase_gate,
    summarize_action_correlations,
    summarize_phase_traces,
)
from v5_6_ppo_cfg import V5_6_PPO_CFG
from v5_7_env_cfg import UsageFloorGuideEnvCfg, V5_7_REWARD_WEIGHTS
from v5_7_math import evaluate_v5_7_phase_gate, summarize_usage_floor_metrics
from v5_7_ppo_cfg import V5_7_PPO_CFG
from v5_8_env_cfg import GatedVelocityGuideEnvCfg, V5_8_REWARD_WEIGHTS
from v5_8_math import evaluate_v5_8_phase_gate, summarize_usage_gate_metrics
from v5_8_ppo_cfg import V5_8_PPO_CFG
from v5_9_env_cfg import SymmetricGatedVelocityGuideEnvCfg, V5_9_REWARD_WEIGHTS
from v5_9_math import evaluate_v5_9_phase_gate
from v5_9_ppo_cfg import V5_9_PPO_CFG
from v5_11_env_cfg import ReferenceResetGuideEnvCfg, V5_11_REWARD_WEIGHTS
from v5_11_ppo_cfg import V5_11_PPO_CFG
from v5_12_env_cfg import AntiCheatGuideEnvCfg, V5_12_REWARD_WEIGHTS
from v5_12_math import evaluate_v5_12_phase_gate
from v5_12_ppo_cfg import V5_12_PPO_CFG
from v5_13_env_cfg import MeasuredAntiCheatGuideEnvCfg, V5_13_REWARD_WEIGHTS
from v5_13_math import evaluate_v5_13_phase_gate
from v5_13_ppo_cfg import V5_13_PPO_CFG
from v5_14_env_cfg import PostureFixGuideEnvCfg, V5_14_REWARD_WEIGHTS
from v5_14_math import evaluate_v5_14_phase_gate
from v5_14_ppo_cfg import V5_14_PPO_CFG
from v6_env_cfg import UprightAntiCheatGuideEnvCfg, V6_REWARD_WEIGHTS
from v6_ppo_cfg import V6_PPO_CFG
from v7_env_cfg import SmallActionGuideEnvCfg, V7_REWARD_WEIGHTS
from v7_ppo_cfg import V7_PPO_CFG
import smooth_mdp


runner_module.SmoothActorCritic = SmoothActorCritic
gym.register(
    id=EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GroundedGuideEnvCfg,
        "rsl_rl_cfg_entry_point": SMOOTH_GUIDE_PPO_CFG,
    },
)
gym.register(
    id=V5_3_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ReferenceTrainingGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_3_PPO_CFG,
    },
)
gym.register(
    id=V5_4_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PatternNeutralGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_4_PPO_CFG,
    },
)
gym.register(
    id=V5_6_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PhaseDiverseGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_6_PPO_CFG,
    },
)
gym.register(
    id=V5_7_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UsageFloorGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_7_PPO_CFG,
    },
)
gym.register(
    id=V5_8_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GatedVelocityGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_8_PPO_CFG,
    },
)
gym.register(
    id=V5_9_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SymmetricGatedVelocityGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_9_PPO_CFG,
    },
)
gym.register(
    id=V5_11_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ReferenceResetGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_11_PPO_CFG,
    },
)
gym.register(
    id=V5_12_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AntiCheatGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_12_PPO_CFG,
    },
)
gym.register(
    id=V5_13_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MeasuredAntiCheatGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V5_13_PPO_CFG,
    },
)
gym.register(
    id=V6_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UprightAntiCheatGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V6_PPO_CFG,
    },
)
gym.register(
    id=V7_EVALUATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SmallActionGuideEnvCfg,
        "rsl_rl_cfg_entry_point": V7_PPO_CFG,
    },
)


def main() -> bool:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    protocol = PROTOCOLS[args_cli.protocol]
    num_envs = protocol["num_envs"]
    if args_cli.variant == "v7":
        env_cfg = SmallActionGuideEnvCfg()
        agent_cfg = copy.deepcopy(V7_PPO_CFG)
        task_id = V7_EVALUATION_TASK_ID
    elif args_cli.variant == "v6":
        env_cfg = UprightAntiCheatGuideEnvCfg()
        agent_cfg = copy.deepcopy(V6_PPO_CFG)
        task_id = V6_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_13":
        env_cfg = MeasuredAntiCheatGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_13_PPO_CFG)
        task_id = V5_13_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_12":
        env_cfg = AntiCheatGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_12_PPO_CFG)
        task_id = V5_12_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_11":
        env_cfg = ReferenceResetGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_11_PPO_CFG)
        task_id = V5_11_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_9":
        env_cfg = SymmetricGatedVelocityGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_9_PPO_CFG)
        task_id = V5_9_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_8":
        env_cfg = GatedVelocityGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_8_PPO_CFG)
        task_id = V5_8_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_7":
        env_cfg = UsageFloorGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_7_PPO_CFG)
        task_id = V5_7_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_6":
        env_cfg = PhaseDiverseGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_6_PPO_CFG)
        task_id = V5_6_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_4":
        env_cfg = PatternNeutralGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_4_PPO_CFG)
        task_id = V5_4_EVALUATION_TASK_ID
    elif args_cli.variant == "v5_3":
        env_cfg = ReferenceTrainingGuideEnvCfg()
        agent_cfg = copy.deepcopy(V5_3_PPO_CFG)
        task_id = V5_3_EVALUATION_TASK_ID
    else:
        env_cfg = GroundedGuideEnvCfg()
        agent_cfg = copy.deepcopy(SMOOTH_GUIDE_PPO_CFG)
        task_id = EVALUATION_TASK_ID
    env_cfg.sim.device = args_cli.device
    eval_speed = protocol["speed"] if args_cli.speed is None else args_cli.speed
    configure_deterministic_evaluation(env_cfg, protocol["num_envs"], eval_speed)
    agent_cfg.device = env_cfg.sim.device
    gym_env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint, load_optimizer=False)
        runner.alg.policy.reset()
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        ordered_sensor = SceneEntityCfg(
            "contact_forces",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            preserve_order=True,
        )
        foot_sensor = SceneEntityCfg("contact_forces", body_names=".*_foot")
        ordered_sensor.resolve(base_env.scene)
        foot_sensor.resolve(base_env.scene)
        contact_sensor = base_env.scene.sensors[ordered_sensor.name]
        action_term = base_env.action_manager.get_term("joint_pos")
        action_joint_names = tuple(action_term._joint_names)
        contact_traces = []
        first_contact_traces = []
        first_air_traces = []
        action_traces = []
        steps = round(protocol["duration"] / base_env.step_dt)
        start_y = robot.data.root_pos_w[:, 1].clone()
        max_lateral = torch.zeros(num_envs, device=base_env.device)
        max_heading = torch.zeros(num_envs, device=base_env.device)
        height_min = torch.full((num_envs,), torch.inf, device=base_env.device)
        height_max = torch.full((num_envs,), -torch.inf, device=base_env.device)
        wrong_streak = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
        max_wrong_streak = torch.zeros_like(wrong_streak)
        under_support_streak = torch.zeros_like(wrong_streak)
        max_under_support_streak = torch.zeros_like(wrong_streak)
        flight_streak = torch.zeros_like(wrong_streak)
        max_flight_streak = torch.zeros_like(wrong_streak)
        alive = torch.ones(num_envs, dtype=torch.bool, device=base_env.device)
        termination_counts = {name: 0 for name in ("time_out", "base_contact", "low_height")}
        sum_forward = sum_height = sum_yaw_sq = sum_vz_sq = sum_rp_sq = sum_impact = 0.0
        support_good = flight_samples = total_samples = 0
        all_terms_finite = True
        if args_cli.variant == "v7":
            required_terms = set(V7_REWARD_WEIGHTS)
        elif args_cli.variant == "v6":
            required_terms = set(V6_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_13":
            required_terms = set(V5_13_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_12":
            required_terms = set(V5_12_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_11":
            required_terms = set(V5_11_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_9":
            required_terms = set(V5_9_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_8":
            required_terms = set(V5_8_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_7":
            required_terms = set(V5_7_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_6":
            required_terms = set(V5_6_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_4":
            required_terms = set(V5_4_REWARD_WEIGHTS)
        elif args_cli.variant == "v5_3":
            required_terms = set(V5_3_REWARD_WEIGHTS)
        else:
            required_terms = {
                "signed_forward_progress",
                "wrong_direction",
                "support_deficit",
                "flight",
                "lateral_velocity",
                "yaw_rate",
                "heading_hold",
                "base_height_centered",
                "vertical_acceleration",
                "foot_impact",
                "foot_slip",
                "swing_clearance",
            }
        all_terms_active = required_terms.issubset(base_env.reward_manager.active_terms)

        for _ in range(steps):
            root_lin_b = robot.data.root_lin_vel_b
            root_ang_b = robot.data.root_ang_vel_b
            forward = smooth_mdp.guide_base_lin_vel(base_env)[:, 0]
            yaw = torch.abs(quaternion_yaw_wxyz(robot.data.root_quat_w))
            contacts = smooth_mdp.foot_contact_flags(base_env, 5.0, ordered_sensor).bool()
            first_contact = contact_sensor.compute_first_contact(base_env.step_dt)[
                :, ordered_sensor.body_ids
            ]
            first_air = contact_sensor.compute_first_air(base_env.step_dt)[
                :, ordered_sensor.body_ids
            ]
            contact_traces.append(contacts.detach().cpu())
            first_contact_traces.append(first_contact.detach().cpu())
            first_air_traces.append(first_air.detach().cpu())
            contact_count = contacts.sum(dim=-1)
            height = smooth_mdp.local_base_height_observation(base_env).squeeze(-1)
            impact = smooth_mdp.foot_impact_normalized(base_env, foot_sensor)
            wrong_streak = torch.where(
                alive,
                torch.where(forward < 0.0, wrong_streak + 1, torch.zeros_like(wrong_streak)),
                wrong_streak,
            )
            max_wrong_streak = torch.maximum(max_wrong_streak, wrong_streak)
            under_support_streak, max_under_support_streak = record_contact_streak(
                under_support_streak,
                max_under_support_streak,
                contact_count < 2.0,
                alive,
            )
            flight_streak, max_flight_streak = record_contact_streak(
                flight_streak,
                max_flight_streak,
                contact_count == 0.0,
                alive,
            )
            lateral = torch.abs(robot.data.root_pos_w[:, 1] - start_y)
            max_lateral = torch.where(alive, torch.maximum(max_lateral, lateral), max_lateral)
            max_heading = torch.where(alive, torch.maximum(max_heading, yaw), max_heading)
            height_min = torch.where(alive, torch.minimum(height_min, height), height_min)
            height_max = torch.where(alive, torch.maximum(height_max, height), height_max)
            sum_forward += forward[alive].sum().item()
            sum_height += height[alive].sum().item()
            sum_yaw_sq += root_ang_b[alive, 2].square().sum().item()
            sum_vz_sq += root_lin_b[alive, 2].square().sum().item()
            sum_rp_sq += root_ang_b[alive, :2].square().sum().item()
            sum_impact += impact[alive].sum().item()
            support_good += (contact_count[alive] >= 2.0).sum().item()
            flight_samples += (contact_count[alive] == 0.0).sum().item()
            total_samples += alive.sum().item()
            with torch.inference_mode():
                actions = policy(obs)
                action_traces.append(actions.detach().cpu())
                obs, _, dones, _ = env.step(actions)
                runner.alg.policy.reset(dones)
            for name in termination_counts:
                term_dones = base_env.termination_manager.get_term(name).bool()
                termination_counts[name] += (alive & term_dones).sum().item()
            all_terms_finite &= torch.isfinite(base_env.reward_manager._step_reward[alive]).all().item()
            alive = advance_alive_mask(alive, dones.bool())

        metrics = {
            "mean_forward_speed_m_s": sum_forward / total_samples,
            "max_wrong_direction_s": max_wrong_streak.max().item() * base_env.step_dt,
            "p95_lateral_displacement_m": torch.quantile(max_lateral, 0.95).item(),
            "p95_heading_drift_deg": math.degrees(torch.quantile(max_heading, 0.95).item()),
            "rms_yaw_rate_rad_s": math.sqrt(sum_yaw_sq / total_samples),
            "support_ge2_fraction": support_good / total_samples,
            "max_under_support_s": max_under_support_streak.max().item() * base_env.step_dt,
            "flight_fraction": flight_samples / total_samples,
            "max_flight_s": max_flight_streak.max().item() * base_env.step_dt,
            "mean_base_height_m": sum_height / total_samples,
            "p95_height_ptp_m": torch.quantile(height_max - height_min, 0.95).item(),
            "rms_vertical_velocity_m_s": math.sqrt(sum_vz_sq / total_samples),
            "rms_roll_pitch_rate_rad_s": math.sqrt(sum_rp_sq / total_samples),
            "mean_normalized_foot_impact": sum_impact / total_samples,
            "survival_fraction": alive.float().mean().item(),
            "termination_counts": termination_counts,
            "all_terms_finite": bool(all_terms_finite),
            "all_terms_active": bool(all_terms_active),
        }
        phase_metrics = summarize_phase_traces(
            torch.stack(contact_traces),
            torch.stack(first_contact_traces),
            torch.stack(first_air_traces),
            foot_names=("FL", "FR", "RL", "RR"),
        )
        phase_metrics.update(
            summarize_action_correlations(torch.stack(action_traces), action_joint_names)
        )
        if args_cli.variant in ("v5_8", "v5_9"):
            phase_metrics.update(summarize_usage_gate_metrics(phase_metrics))
        elif args_cli.variant == "v5_7":
            phase_metrics.update(summarize_usage_floor_metrics(phase_metrics))
        gate_fn = evaluate_pilot_gate if args_cli.protocol == "pilot" else evaluate_gate
        body_gates = gate_fn(metrics)
        if args_cli.variant in ("v7", "v6", "v5_13"):
            phase_gates = evaluate_v5_13_phase_gate(phase_metrics)
        elif args_cli.variant == "v5_12":
            phase_gates = evaluate_v5_12_phase_gate(phase_metrics)
        elif args_cli.variant == "v5_9":
            phase_gates = evaluate_v5_9_phase_gate(phase_metrics)
        elif args_cli.variant == "v5_8":
            phase_gates = evaluate_v5_8_phase_gate(phase_metrics)
        elif args_cli.variant == "v5_7":
            phase_gates = evaluate_v5_7_phase_gate(phase_metrics)
        elif args_cli.variant == "v5_6":
            phase_gates = evaluate_phase_gate(phase_metrics)
        else:
            phase_gates = {}
        payload = {"metrics": metrics, "phase_metrics": phase_metrics}
        payload["body_gates"] = body_gates
        payload["phase_gates"] = phase_gates
        if args_cli.variant in ("v5_6", "v5_7", "v5_8", "v5_9", "v5_12", "v5_13", "v6", "v7"):
            gates = {**body_gates, **phase_gates}
        else:
            gates = body_gates
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
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
