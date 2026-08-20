"""Measure V11 climbing on ascending stairs spanning 5--20 cm steps."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from evaluation_cli import add_seed_argument


TASK_ID = "Isaac-Velocity-Stairs-Guide-RobotV2-Eval-v11"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="v11_stairs_gate.json")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--env", choices=("v11", "v10_4"), default="v11")
add_seed_argument(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from evaluation_cli import evaluation_metadata, load_inference_checkpoint
from gait_phase_metrics import contact_phase_metrics
from v9_env_cfg import (
    STAIRS_NUM_LEVELS,
    STAIRS_STEP_HEIGHT_RANGE,
    STAIRS_STEP_WIDTH,
    STAIRS_UP_EVAL_TERRAIN_CFG,
)
from v9_eval_metrics import climbed_stairs
from v11_eval_metrics import (
    MIN_REALISTIC_STEP_HEIGHT_M,
    minimum_level_for_step_height,
    peak_height_after_approach,
    stair_level_assignments,
    stairs_gates,
)
from v11_terrain_env_cfg import GuideV2ScannerTerrainEnvCfg
from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG
from v10_4_terrain_env_cfg import GuideV2BlindTerrainEnvCfg
from v8_ppo_cfg import V8_PPO_CFG


if args_cli.env == "v10_4":
    EVAL_ENV_CFG = GuideV2BlindTerrainEnvCfg
    EVAL_PPO_CFG = V8_PPO_CFG
else:
    EVAL_ENV_CFG = GuideV2ScannerTerrainEnvCfg
    EVAL_PPO_CFG = V11_TERRAIN_PPO_CFG


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EVAL_ENV_CFG,
        "rsl_rl_cfg_entry_point": EVAL_PPO_CFG,
    },
)

FEET = ("FL", "FR", "RL", "RR")
STAIRS_PLATFORM_WIDTH_M = 3.0
STAIRS_APPROACH_DISTANCE_M = (
    STAIRS_PLATFORM_WIDTH_M / 2.0 + 2.0 * STAIRS_STEP_WIDTH
)
REALISTIC_STEP_HEIGHT_M = MIN_REALISTIC_STEP_HEIGHT_M


def _set_stair_levels(raw_env) -> None:
    """Pin equal groups of robots to each difficulty row before evaluation."""
    assignments = stair_level_assignments(args_cli.num_envs, STAIRS_NUM_LEVELS)
    base_env = raw_env.unwrapped
    terrain = base_env.scene.terrain
    terrain.terrain_levels[:] = torch.tensor(
        assignments,
        device=base_env.device,
        dtype=terrain.terrain_levels.dtype,
    )
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels,
        terrain.terrain_types,
    ]
    raw_env.reset()


def main() -> bool:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)

    env_cfg = EVAL_ENV_CFG()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 60.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.terrain.terrain_generator = STAIRS_UP_EVAL_TERRAIN_CFG
    env_cfg.scene.terrain.max_init_terrain_level = STAIRS_NUM_LEVELS - 1
    env_cfg.curriculum = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        key: (0.0, 0.0)
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (60.0, 60.0)
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (args_cli.speed, args_cli.speed)
    command.ranges.ang_vel_z = (0.0, 0.0)

    agent_cfg = copy.deepcopy(EVAL_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    _set_stair_levels(raw_env)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        load_inference_checkpoint(runner, args_cli.checkpoint)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        terrain = base_env.scene.terrain
        levels = terrain.terrain_levels.clone().cpu()

        foot_sensor = SceneEntityCfg(
            "contact_forces",
            body_names=[f"{foot}_foot" for foot in FEET],
            preserve_order=True,
        )
        foot_sensor.resolve(base_env.scene)
        sensor = base_env.scene.sensors[foot_sensor.name]

        steps = int(args_cli.seconds / base_env.step_dt)
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        contacts = []
        speeds = []
        valid_samples = []
        height_gain_samples = []
        horizontal_progress_samples = []
        alive = torch.ones(
            args_cli.num_envs,
            dtype=torch.bool,
            device=base_env.device,
        )
        start_xy = robot.data.root_pos_w[:, :2].clone()
        start_height = robot.data.root_pos_w[:, 2].clone()
        with torch.inference_mode():
            for _ in range(steps):
                valid_samples.append(alive.cpu())
                height_gain_samples.append(
                    (robot.data.root_pos_w[:, 2] - start_height).cpu()
                )
                horizontal_progress_samples.append(
                    torch.linalg.vector_norm(
                        robot.data.root_pos_w[:, :2] - start_xy,
                        dim=-1,
                    ).cpu()
                )
                forces = sensor.data.net_forces_w_history[
                    :, :, foot_sensor.body_ids
                ]
                contacts.append(
                    (forces.norm(dim=-1).max(dim=1)[0] > 5.0).float().cpu()
                )
                speeds.append(robot.data.root_lin_vel_b[:, 1].cpu())
                action = policy(obs)
                step = env.step(action)
                obs, dones = step[0], step[2]
                alive &= ~dones.bool()

        contact_t = torch.stack(contacts)
        valid_t = torch.stack(valid_samples)
        speed_t = torch.stack(speeds)
        height_gain_t = torch.stack(height_gain_samples)
        horizontal_progress_t = torch.stack(horizontal_progress_samples)
        phase = contact_phase_metrics(contact_t)
        direct_valid_fraction = min(
            pair["valid_fraction"] for pair in phase["pairs"].values()
        )
        height_gain = (robot.data.root_pos_w[:, 2] - start_height).cpu()
        horizontal_progress = torch.linalg.vector_norm(
            robot.data.root_pos_w[:, :2] - start_xy,
            dim=-1,
        ).cpu()
        peak_height_gain = peak_height_after_approach(
            height_gain_t,
            horizontal_progress_t,
            valid_t,
            STAIRS_APPROACH_DISTANCE_M,
        )
        alive_cpu = alive.cpu()
        low, high = STAIRS_STEP_HEIGHT_RANGE
        span = STAIRS_NUM_LEVELS - 1
        step_height = low + (high - low) * levels.float() / span
        climbed = climbed_stairs(
            peak_height_gain,
            horizontal_progress,
            step_height,
            STAIRS_APPROACH_DISTANCE_M,
        ) & alive_cpu

        per_level = {}
        raw_climb_success_by_level = {}
        for level in range(STAIRS_NUM_LEVELS):
            mask = levels == level
            raw_climb_success = float(climbed[mask].float().mean())
            raw_climb_success_by_level[level] = raw_climb_success
            per_level[str(level)] = {
                "step_height_m": round(float(low + (high - low) * level / span), 3),
                "n": int(mask.sum()),
                "survival": round(float(alive_cpu[mask].float().mean()), 3),
                "climb_success": round(raw_climb_success, 3),
                "mean_height_gain_m": round(float(height_gain[mask].mean()), 3),
                "mean_peak_height_gain_m": round(
                    float(peak_height_gain[mask].mean()), 3
                ),
                "mean_horizontal_progress_m": round(
                    float(horizontal_progress[mask].mean()), 3
                ),
            }
        hardest_level_cleared = max(
            (
                level
                for level, success in raw_climb_success_by_level.items()
                if success >= 0.5
            ),
            default=-1,
        )
        hardest_step_height_cleared_m = (
            low + (high - low) * hardest_level_cleared / span
            if hardest_level_cleared >= 0
            else 0.0
        )
        stairs = {
            "step_height_range_m": list(STAIRS_STEP_HEIGHT_RANGE),
            "minimum_approach_distance_m": STAIRS_APPROACH_DISTANCE_M,
            "climb_success_fraction": float(climbed.float().mean()),
            "hardest_level_cleared": hardest_level_cleared,
            "hardest_step_height_cleared_m": hardest_step_height_cleared_m,
            "per_level": per_level,
        }
        metrics = {
            "survival_fraction": float(alive.float().mean()),
            "mean_forward_speed_m_s": float(speed_t[valid_t].mean()),
            "direct_phase_valid_fraction": direct_valid_fraction,
            "commanded_speed": args_cli.speed,
            "seconds": args_cli.seconds,
            "scenario": "stairs_up",
            "environment": args_cli.env,
        }
        realistic_level = minimum_level_for_step_height(
            STAIRS_STEP_HEIGHT_RANGE,
            STAIRS_NUM_LEVELS,
            REALISTIC_STEP_HEIGHT_M,
        )
        gates = stairs_gates(
            metrics,
            stairs,
            minimum_step_height_m=REALISTIC_STEP_HEIGHT_M,
        )
        payload = {
            "metadata": evaluation_metadata(args_cli.checkpoint, args_cli.seed),
            "metrics": metrics,
            "phase_metrics": {"contact_phase": phase},
            "stairs": stairs,
            "realistic_step": {
                "minimum_height_m": REALISTIC_STEP_HEIGHT_M,
                "first_level": realistic_level,
            },
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
