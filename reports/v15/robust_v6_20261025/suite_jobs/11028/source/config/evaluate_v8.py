"""Evaluate a V8 checkpoint and report the gait diagnostics used since V5.13.

Reports the same quantities the V5.11-V7 comparisons used -- per-foot contact
fractions, HFE same-pair correlations, forward speed, base height, survival --
so V8 can be placed directly against those runs.
"""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from evaluation_cli import (
    add_fixed_terrain_level_argument,
    add_seed_argument,
    evaluation_metadata,
    load_inference_checkpoint,
    validate_fixed_terrain_level,
)


TASK_ID = "Isaac-Velocity-Flat-Guide-Spot-Recipe-Eval-v8"

parser = argparse.ArgumentParser(description="Evaluate a V8 Spot-recipe checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="v8_gate.json")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--seconds", type=float, default=8.0)
# Which robot/config the checkpoint was trained with. Survival is a gate and
# the termination set decides it, so the environment must match.
parser.add_argument("--env", choices=("v8", "v10_flat", "v10_1_flat", "v10_2_flat", "v10_3_flat", "v10_4_terrain", "v11_terrain"), default="v8")
# Which base-frame axis the robot's head points along. The v1 robot faced
# base -x; the v2 robot's fore-aft axis is y (front and rear foot pairs are
# 0.88 m apart in y, 0.52 m in x, and the whole-robot extent is 1.42 m in y
# against 0.80 m in x). Scoring a +y policy under the -x convention reports a
# forward speed of roughly zero.
parser.add_argument("--forward-axis", choices=("-x", "+y"), default="-x")
# Seconds of rollout discarded before metrics start accumulating. Default 0.0, so
# every archived gate number is reproduced exactly. It exists because the robot
# spawned 0.81 m underground until V10.3 and depenetration took ~200 steps under
# zero action, which this evaluator folded into its contact fractions.
parser.add_argument("--warmup-seconds", type=float, default=0.0)
add_fixed_terrain_level_argument(parser)
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

from v8_env_cfg import GuideSpotRecipeEnvCfg
from v8_ppo_cfg import V8_PPO_CFG

if args_cli.env == "v11_terrain":
    from v11_terrain_env_cfg import GuideV2ScannerTerrainEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v10_4_terrain":
    from v10_4_terrain_env_cfg import GuideV2BlindTerrainEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v10_3_flat":
    from v10_3_flat_env_cfg import GuideV2FlatSpawnFixedEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v10_2_flat":
    from v10_2_flat_env_cfg import GuideV2FlatDutyLiveEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v10_1_flat":
    from v10_1_flat_env_cfg import GuideV2FlatForwardEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v10_flat":
    from v10_flat_env_cfg import GuideV2FlatEnvCfg as EVAL_ENV_CFG
else:
    EVAL_ENV_CFG = GuideSpotRecipeEnvCfg


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EVAL_ENV_CFG,
        "rsl_rl_cfg_entry_point": V8_PPO_CFG,
    },
)

FEET = ("FL", "FR", "RL", "RR")
PAIRS = (("FL", "FR"), ("RL", "RR"), ("FL", "RR"), ("FR", "RL"), ("FL", "RL"), ("FR", "RR"))


def main() -> bool:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    import copy

    env_cfg = EVAL_ENV_CFG()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 60.0
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.terrain_level is not None:
        generator = env_cfg.scene.terrain.terrain_generator
        if generator is None:
            raise ValueError("--terrain-level requires generated terrain")
        validate_fixed_terrain_level(
            args_cli.terrain_level,
            int(generator.num_rows),
        )
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.terrain_level
        env_cfg.curriculum = None
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (60.0, 60.0)
    cmd.rel_standing_envs = 0.0
    # --speed is a guide-frame forward speed; the command lives in the base frame,
    # where the robot's head points along -x.
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
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}

    agent_cfg = copy.deepcopy(V8_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    if args_cli.terrain_level is not None:
        terrain = raw_env.unwrapped.scene.terrain
        terrain.terrain_levels.fill_(args_cli.terrain_level)
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels,
            terrain.terrain_types,
        ]
        raw_env.reset()
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        load_inference_checkpoint(runner, args_cli.checkpoint)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        foot_sensor = SceneEntityCfg("contact_forces", body_names=list(f"{f}_foot" for f in FEET), preserve_order=True)
        foot_sensor.resolve(base_env.scene)
        hfe = SceneEntityCfg("robot", joint_names=[f"{f}_HFE" for f in FEET], preserve_order=True)
        hfe.resolve(base_env.scene)
        sensor = base_env.scene.sensors[foot_sensor.name]

        steps = int(args_cli.seconds / base_env.step_dt)
        warmup_steps = int(args_cli.warmup_seconds / base_env.step_dt)
        if warmup_steps >= steps:
            raise SystemExit(f"--warmup-seconds {args_cli.warmup_seconds} leaves no rollout")
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        contacts, hfe_vel, speeds, heights = [], [], [], []
        alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device=base_env.device)
        with torch.inference_mode():
            for _step in range(steps):
                # Survival still accrues from step 0; only the gait metrics skip
                # the warmup, because a fall during it is a real fall.
                if _step >= warmup_steps:
                    forces = sensor.data.net_forces_w_history[:, :, foot_sensor.body_ids]
                    contact = (forces.norm(dim=-1).max(dim=1)[0] > 5.0).float()
                    contacts.append(contact.cpu())
                    hfe_vel.append(robot.data.joint_vel[:, hfe.joint_ids].cpu())
                    # Report guide-frame forward speed (positive = head first).
                    if args_cli.forward_axis == "+y":
                        speeds.append(robot.data.root_lin_vel_b[:, 1].cpu())
                    else:
                        speeds.append(-robot.data.root_lin_vel_b[:, 0].cpu())
                    heights.append(robot.data.root_pos_w[:, 2].cpu())
                action = policy(obs)
                step = env.step(action)
                obs, dones = step[0], step[2]
                alive &= ~dones.bool()

        contact_t = torch.stack(contacts)
        vel_t = torch.stack(hfe_vel)
        from gait_phase_metrics import contact_phase_metrics

        direct_contact_phase = contact_phase_metrics(contact_t)
        contact_fraction = {f: float(contact_t[..., i].mean()) for i, f in enumerate(FEET)}
        swing_fraction = {f: 1.0 - v for f, v in contact_fraction.items()}
        idx = {f: i for i, f in enumerate(FEET)}
        corr = {}
        flat = vel_t.reshape(-1, 4)
        centered = flat - flat.mean(dim=0, keepdim=True)
        norms = centered.norm(dim=0)
        for a, b in PAIRS:
            denom = (norms[idx[a]] * norms[idx[b]]).clamp(min=1e-8)
            corr[f"{a}-{b}"] = float((centered[:, idx[a]] * centered[:, idx[b]]).sum() / denom)

        payload = {
            "metadata": evaluation_metadata(args_cli.checkpoint, args_cli.seed),
            "metrics": {
                "mean_forward_speed_m_s": float(torch.stack(speeds).mean()),
                "mean_base_height_m": float(torch.stack(heights).mean()),
                "survival_fraction": float(alive.float().mean()),
                "commanded_speed": args_cli.speed,
                "forward_axis": args_cli.forward_axis,
                "warmup_seconds": args_cli.warmup_seconds,
                "terrain_level": args_cli.terrain_level,
            },
            "phase_metrics": {
                "foot_contact_fraction": contact_fraction,
                "foot_swing_fraction": swing_fraction,
                "min_foot_contact_fraction": min(contact_fraction.values()),
                "min_foot_swing_fraction": min(swing_fraction.values()),
                "same_joint_pair_correlation": {"HFE": corr},
                "contact_phase": direct_contact_phase,
            },
        }
        # A trot: diagonals in phase (positive), same-side pairs out of phase.
        gates = {
            "front_pair_out_of_phase": corr["FL-FR"] <= 0.30,
            "rear_pair_out_of_phase": corr["RL-RR"] <= 0.30,
            "diagonal_FL_RR_in_phase": corr["FL-RR"] >= 0.0,
            "diagonal_FR_RL_in_phase": corr["FR-RL"] >= 0.0,
            "min_foot_contact": min(contact_fraction.values()) >= 0.30,
            "min_foot_swing": min(swing_fraction.values()) >= 0.20,
            "forward_speed": payload["metrics"]["mean_forward_speed_m_s"] >= 0.6 * args_cli.speed,
            "survival": payload["metrics"]["survival_fraction"] >= 0.95,
        }
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        Path(args_cli.output).write_text(json.dumps(payload, indent=2) + "\n")
        for name, ok in gates.items():
            print(f"[GATE] {name}={'PASS' if ok else 'FAIL'}")
        return payload["passed"]
    finally:
        env.close()


if __name__ == "__main__":
    passed = main()
    simulation_app.close()
    raise SystemExit(0 if passed else 1)
