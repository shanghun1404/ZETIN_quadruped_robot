"""Evaluate a V9 checkpoint and report the gait diagnostics used since V5.13.

Reports the same quantities the V5.11-V7 comparisons used -- per-foot contact
fractions, HFE same-pair correlations, forward speed, base height, survival --
so V8 can be placed directly against those runs.
"""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Rough-Guide-Terrain-Eval-v9"

parser = argparse.ArgumentParser(description="Evaluate a V9 terrain checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="v9_gate.json")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument(
    "--scenario",
    choices=("rough", "stairs_up"),
    default="rough",
    help="rough = the training terrain mix; stairs_up = stairs only, measured.",
)
# Survival is a *reported gate*, and it is decided by the termination set, so a
# V9.1 checkpoint scored under V9's stricter set would be judged for contacts its
# own training deliberately tolerates. The environment must match what trained.
parser.add_argument(
    "--env",
    choices=("v9", "v9_1", "v9_2", "v9_3", "v9_4", "v10", "v10_1", "v10_2"),
    default="v10_2",
    help="which terrain config to evaluate under; must match the checkpoint's training config.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from v9_env_cfg import (
    GuideTerrainEnvCfg,
    STAIRS_NUM_LEVELS,
    STAIRS_STEP_WIDTH,
    STAIRS_STEP_HEIGHT_RANGE,
    STAIRS_UP_EVAL_TERRAIN_CFG,
)
from v9_eval_metrics import climbed_stairs, pearson_correlation_or_none
from v9_ppo_cfg import V9_PPO_CFG

if args_cli.env == "v10_2":
    from v10_2_env_cfg import GuideV2TurnBandEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v10_1":
    from v10_1_env_cfg import GuideV2BaselineEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v10":
    from v10_env_cfg import GuideRobotV2EnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v9_4":
    from v9_4_env_cfg import GuideTerrainTravelGatedEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v9_3":
    from v9_3_env_cfg import GuideTerrainSharpVelEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v9_2":
    from v9_2_env_cfg import GuideTerrainTurnBandEnvCfg as EVAL_ENV_CFG
elif args_cli.env == "v9_1":
    from v9_1_env_cfg import GuideTerrainFixedEnvCfg as EVAL_ENV_CFG
else:
    EVAL_ENV_CFG = GuideTerrainEnvCfg


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EVAL_ENV_CFG,
        "rsl_rl_cfg_entry_point": V9_PPO_CFG,
    },
)

FEET = ("FL", "FR", "RL", "RR")
PAIRS = (("FL", "FR"), ("RL", "RR"), ("FL", "RR"), ("FR", "RL"), ("FL", "RL"), ("FR", "RR"))

# The inverted-pyramid terrain starts on a 3.0 m-wide bottom platform. Requiring
# half that width plus two 0.30 m treads prevents a vertical hop in the pit from
# being reported as a stair climb. V9.2 gained 0.000 m on stairs; the metric must
# continue to describe travel up the structure, not just transient body motion.
STAIRS_PLATFORM_WIDTH_M = 3.0
STAIRS_APPROACH_DISTANCE_M = STAIRS_PLATFORM_WIDTH_M / 2.0 + 2.0 * STAIRS_STEP_WIDTH


def _base_clearance_m(base_env, robot):
    """Base height above the nearest scanner hit, not world-frame altitude."""
    hits = base_env.scene.sensors["height_scanner"].data.ray_hits_w
    root_xy = robot.data.root_pos_w[:, None, :2]
    horizontal_distance_sq = (hits[..., :2] - root_xy).square().sum(dim=-1)
    finite = torch.isfinite(hits).all(dim=-1)
    horizontal_distance_sq = horizontal_distance_sq.masked_fill(~finite, float("inf"))
    nearest = horizontal_distance_sq.argmin(dim=1)
    ground_z = hits[..., 2].gather(1, nearest.unsqueeze(1)).squeeze(1)
    # RayCaster returns inf for a ray that hits no mesh, which happens at terrain
    # borders. Raising here would abort a whole evaluation over one bad ray and,
    # because the queue runs evaluators under `|| true`, would leave no JSON at
    # all. Fall back to the env origin's ground plane for those robots and report
    # how many needed it, so a systematically blind scanner is still visible.
    missing = ~torch.isfinite(ground_z)
    if bool(missing.any()):
        fallback = base_env.scene.env_origins[:, 2]
        ground_z = torch.where(missing, fallback, ground_z)
    return robot.data.root_pos_w[:, 2] - ground_z, int(missing.sum())


def _set_stair_levels(raw_env) -> None:
    """Assign every stair difficulty deterministically before the evaluation reset."""
    if args_cli.num_envs < STAIRS_NUM_LEVELS:
        raise ValueError(
            f"stairs_up needs at least {STAIRS_NUM_LEVELS} environments to measure every level"
        )
    terrain = raw_env.unwrapped.scene.terrain
    levels = torch.arange(args_cli.num_envs, device=raw_env.unwrapped.device)
    terrain.terrain_levels[:] = levels.remainder(STAIRS_NUM_LEVELS)
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types
    ]
    # reset_base reads the updated environment origins. Without this reset the
    # level labels change while the robots remain on their old, random tiles.
    raw_env.reset()


def main() -> bool:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    import copy

    env_cfg = EVAL_ENV_CFG()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    env_cfg.episode_length_s = 60.0
    env_cfg.observations.policy.enable_corruption = False
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (60.0, 60.0)
    cmd.rel_standing_envs = 0.0
    # --speed is a guide-frame forward speed; the command lives in the base frame,
    # where the robot's head points along -x.
    from guide_frame import guide_forward_speed_range_to_base

    cmd.ranges.lin_vel_x = guide_forward_speed_range_to_base((args_cli.speed, args_cli.speed))
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    if args_cli.scenario == "stairs_up":
        # Stairs only, spread across every difficulty row so one run reports the
        # tallest step the policy clears.
        env_cfg.scene.terrain.terrain_generator = STAIRS_UP_EVAL_TERRAIN_CFG
        env_cfg.scene.terrain.max_init_terrain_level = STAIRS_NUM_LEVELS - 1
        # Freeze the curriculum: a fixed level per environment is what makes the
        # per-level success rate meaningful.
        env_cfg.curriculum = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}

    agent_cfg = copy.deepcopy(V9_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    if args_cli.scenario == "stairs_up":
        _set_stair_levels(raw_env)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        foot_sensor = SceneEntityCfg("contact_forces", body_names=list(f"{f}_foot" for f in FEET), preserve_order=True)
        foot_sensor.resolve(base_env.scene)
        hfe = SceneEntityCfg("robot", joint_names=[f"{f}_HFE" for f in FEET], preserve_order=True)
        hfe.resolve(base_env.scene)
        sensor = base_env.scene.sensors[foot_sensor.name]

        steps = int(args_cli.seconds / base_env.step_dt)
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        contacts, hfe_vel, speeds, heights, valid_samples = [], [], [], [], []
        scanner_misses = 0
        start_xy = robot.data.root_pos_w[:, :2].clone()
        start_height = robot.data.root_pos_w[:, 2].clone()
        terrain = base_env.scene.terrain
        levels = (
            terrain.terrain_levels.clone()
            if getattr(terrain, "terrain_levels", None) is not None
            else torch.zeros(args_cli.num_envs, dtype=torch.long, device=base_env.device)
        )
        alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device=base_env.device)
        with torch.inference_mode():
            for _ in range(steps):
                # IsaacLab auto-resets a completed environment. Retaining samples
                # from that new episode would mix a reset transient into the gait
                # that just failed, so all rollout metrics use the first episode.
                valid_samples.append(alive.cpu())
                forces = sensor.data.net_forces_w_history[:, :, foot_sensor.body_ids]
                contact = (forces.norm(dim=-1).max(dim=1)[0] > 5.0).float()
                contacts.append(contact.cpu())
                hfe_vel.append(robot.data.joint_vel[:, hfe.joint_ids].cpu())
                # Report guide-frame forward speed (positive = head first).
                speeds.append(-robot.data.root_lin_vel_b[:, 0].cpu())
                clearance, missed = _base_clearance_m(base_env, robot)
                scanner_misses += missed
                heights.append(clearance.cpu())
                action = policy(obs)
                step = env.step(action)
                obs, dones = step[0], step[2]
                alive &= ~dones.bool()

        contact_t = torch.stack(contacts)
        vel_t = torch.stack(hfe_vel)
        valid_t = torch.stack(valid_samples)
        contact_fraction = {
            f: float(contact_t[..., i][valid_t].mean()) for i, f in enumerate(FEET)
        }
        swing_fraction = {f: 1.0 - v for f, v in contact_fraction.items()}
        idx = {f: i for i, f in enumerate(FEET)}
        corr = {}
        flat = vel_t[valid_t]
        centered = flat - flat.mean(dim=0, keepdim=True)
        norms = centered.norm(dim=0)
        for a, b in PAIRS:
            dot = float((centered[:, idx[a]] * centered[:, idx[b]]).sum())
            corr[f"{a}-{b}"] = pearson_correlation_or_none(
                dot, float(norms[idx[a]]), float(norms[idx[b]])
            )

        height_gain = (robot.data.root_pos_w[:, 2] - start_height).cpu()
        horizontal_progress = torch.linalg.vector_norm(
            robot.data.root_pos_w[:, :2] - start_xy, dim=-1
        ).cpu()
        alive_cpu = alive.cpu()
        speed_t = torch.stack(speeds)
        height_t = torch.stack(heights)
        payload = {
            "metrics": {
                "mean_forward_speed_m_s": float(speed_t[valid_t].mean()),
                "mean_base_height_m": float(height_t[valid_t].mean()),
                "base_height_reference": "nearest finite height-scanner ground hit",
                # Non-zero means some robots fell back to their env-origin ground
                # plane because every scanner ray missed. A large count makes the
                # height number unreliable rather than merely approximate.
                "base_height_scanner_misses": scanner_misses,
                "survival_fraction": float(alive.float().mean()),
                "commanded_speed": args_cli.speed,
                "scenario": args_cli.scenario,
            },
            "phase_metrics": {
                "foot_contact_fraction": contact_fraction,
                "foot_swing_fraction": swing_fraction,
                "min_foot_contact_fraction": min(contact_fraction.values()),
                "min_foot_swing_fraction": min(swing_fraction.values()),
                "same_joint_pair_correlation": {"HFE": corr},
            },
        }
        if args_cli.scenario == "stairs_up":
            low, high = STAIRS_STEP_HEIGHT_RANGE
            span = max(STAIRS_NUM_LEVELS - 1, 1)
            step_h = low + (high - low) * levels.float().cpu() / span
            # Two steps of height alone calls a hop in the 3.0 m-wide starting pit
            # a climb. Require enough horizontal travel to reach two actual treads.
            climbed = climbed_stairs(
                height_gain, horizontal_progress, step_h, STAIRS_APPROACH_DISTANCE_M
            ) & alive_cpu
            per_level = {}
            for lvl in range(STAIRS_NUM_LEVELS):
                mask = (levels.cpu() == lvl)
                if bool(mask.any()):
                    per_level[str(lvl)] = {
                        "step_height_m": round(float(low + (high - low) * lvl / span), 3),
                        "n": int(mask.sum()),
                        "climb_success": round(float(climbed[mask].float().mean()), 3),
                        "mean_height_gain_m": round(float(height_gain[mask].mean()), 3),
                        "mean_horizontal_progress_m": round(
                            float(horizontal_progress[mask].mean()), 3
                        ),
                    }
            payload["stairs"] = {
                "mean_height_gain_m": float(height_gain.mean()),
                "mean_horizontal_progress_m": float(horizontal_progress.mean()),
                "minimum_approach_distance_m": STAIRS_APPROACH_DISTANCE_M,
                "climb_success_fraction": float(climbed.float().mean()),
                "per_level": per_level,
                "hardest_level_cleared": max(
                    (int(k) for k, v in per_level.items() if v["climb_success"] >= 0.5),
                    default=-1,
                ),
            }

        # A trot: diagonals in phase (positive), same-side pairs out of phase.
        # A constant HFE signal has undefined phase. The old denominator clamp
        # turned all four undefined correlations into 0.0, so a motionless policy
        # passed every phase gate -- exactly the V9.2 failure mode.
        gates = {
            "front_pair_out_of_phase": corr["FL-FR"] is not None and corr["FL-FR"] <= 0.30,
            "rear_pair_out_of_phase": corr["RL-RR"] is not None and corr["RL-RR"] <= 0.30,
            "diagonal_FL_RR_in_phase": corr["FL-RR"] is not None and corr["FL-RR"] >= 0.0,
            "diagonal_FR_RL_in_phase": corr["FR-RL"] is not None and corr["FR-RL"] >= 0.0,
            "min_foot_contact": min(contact_fraction.values()) >= 0.30,
            "min_foot_swing": min(swing_fraction.values()) >= 0.20,
            "forward_speed": payload["metrics"]["mean_forward_speed_m_s"] >= 0.6 * args_cli.speed,
            "survival": payload["metrics"]["survival_fraction"] >= 0.95,
        }
        if args_cli.scenario == "stairs_up":
            # On stairs the gait shape is allowed to break; what is graded is
            # whether the robot actually gets up the steps and stays alive.
            gates = {
                "stairs_survival": payload["metrics"]["survival_fraction"] >= 0.80,
                "stairs_climb_success": payload["stairs"]["climb_success_fraction"] >= 0.50,
                "stairs_clears_a_real_step": payload["stairs"]["hardest_level_cleared"] >= 4,
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
