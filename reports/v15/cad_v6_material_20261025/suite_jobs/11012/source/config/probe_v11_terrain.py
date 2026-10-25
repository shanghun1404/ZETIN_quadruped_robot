"""Fail-fast physical and observation probe for V11 scanner terrain."""

from __future__ import annotations

import argparse
import json
import traceback

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Rough-Guide-RobotV2-Probe-v11-terrain"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="run_control/probe_v11_terrain.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from v10_4_terrain_env_cfg import V10_4_REWARD_WEIGHTS
from v11_layout import scan_quality
from v11_terrain_env_cfg import (
    SCAN_POINTS,
    SCANNER_LAYOUT,
    TERRAIN_SPAWN_ROOT_FRAME_Z_M,
    V11_REWARD_WEIGHTS,
    GuideV2ScannerTerrainEnvCfg,
)
from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2ScannerTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": V11_TERRAIN_PPO_CFG,
    },
)

report = {"ok": False}
failures = []
env = None
try:
    cfg = GuideV2ScannerTerrainEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    active_rewards = sorted(base.reward_manager.active_terms)
    report["active_rewards"] = active_rewards
    if active_rewards != sorted(V10_4_REWARD_WEIGHTS):
        failures.append("V11 reward names drifted from V10.4")
    weights = {
        name: float(base.reward_manager.get_term_cfg(name).weight)
        for name in active_rewards
    }
    report["reward_weights"] = weights
    if weights != V11_REWARD_WEIGHTS:
        failures.append("V11 reward weights drifted from V10.4")

    observation_terms = list(base.observation_manager.active_terms["policy"])
    report["observation_terms"] = observation_terms
    if not observation_terms or observation_terms[-1] != "height_scan":
        failures.append("height_scan must be the final policy term for checkpoint expansion")
    observations = base.observation_manager.compute()["policy"]
    report["observation_width"] = int(observations.shape[-1])
    if report["observation_width"] != 48 + SCAN_POINTS:
        failures.append(
            f"observation width {report['observation_width']} != {48 + SCAN_POINTS}"
        )

    scanner = base.scene.sensors.get("height_scanner")
    report["scanner_present"] = scanner is not None
    if scanner is None:
        failures.append("height_scanner is missing")
    else:
        report["scanner_rays"] = int(scanner.data.ray_hits_w.shape[1])
        if report["scanner_rays"] != SCAN_POINTS:
            failures.append(f"scanner rays {report['scanner_rays']} != {SCAN_POINTS}")
    report["scanner_offset"] = list(base.cfg.scene.height_scanner.offset.pos)
    if tuple(report["scanner_offset"]) != tuple(SCANNER_LAYOUT["offset"]):
        failures.append("scanner does not point ahead along base +y")

    report["terrain_type"] = str(base.cfg.scene.terrain.terrain_type)
    if report["terrain_type"] != "generator":
        failures.append("generated terrain is not active")
    curriculum_terms = list(base.curriculum_manager.active_terms)
    report["curriculum_terms"] = curriculum_terms
    if "terrain_levels" not in curriculum_terms:
        failures.append("terrain_levels curriculum is missing")

    command = base.command_manager.get_term("base_velocity").cfg.ranges
    report["lin_vel_x"] = list(command.lin_vel_x)
    report["lin_vel_y"] = list(command.lin_vel_y)
    report["ang_vel_z"] = list(command.ang_vel_z)
    if tuple(command.lin_vel_x) != (0.0, 0.0):
        failures.append("lateral x command is not zero")
    if min(command.lin_vel_y) <= 0.0:
        failures.append("forward command does not point along base +y")
    if tuple(command.ang_vel_z) != (-0.2, 0.2):
        failures.append("terrain yaw band is not [-0.2, 0.2]")

    robot = base.scene["robot"]
    masses = robot.root_physx_view.get_masses()
    mass = float(masses[0].sum())
    report["mass_kg"] = mass
    if not 43.0 < mass < 50.0:
        failures.append(f"unexpected robot mass {mass}")
    terminations = list(base.termination_manager.active_terms)
    report["terminations"] = terminations
    if "base_tilt" not in terminations or "body_contact" in terminations:
        failures.append("robot_v2 orientation termination contract is broken")

    origins_z = base.scene.env_origins[:, 2]
    relative_root_z = robot.data.root_pos_w[:, 2] - origins_z
    report["relative_root_z_before_step"] = float(relative_root_z.mean())
    if abs(report["relative_root_z_before_step"] - TERRAIN_SPAWN_ROOT_FRAME_Z_M) > 0.015:
        failures.append("terrain spawn root height does not match corrected arithmetic")

    zero_action = torch.zeros((args_cli.num_envs, 12), device=base.device)
    env.step(zero_action)
    contact_sensor = base.scene.sensors["contact_forces"]
    force = contact_sensor.data.net_forces_w.norm(dim=-1).sum(dim=1)
    support = float(force.mean()) / (mass * 9.81)
    report["support_on_first_step"] = support
    if support > 0.5:
        failures.append(f"first-step support {support:.3f} indicates terrain penetration")

    for _ in range(40):
        env.step(zero_action)
    scan = base.observation_manager.compute()["policy"][:, -SCAN_POINTS:]
    quality = scan_quality(scan, clip=1.0)
    report["scan_quality"] = quality
    report["scan_std"] = float(scan.std())
    if quality["finite_fraction"] != 1.0:
        failures.append("height scan contains non-finite values")
    if quality["low_saturation_fraction"] > 0.01:
        failures.append("height scan is saturated at the low clip")
    if quality["high_saturation_fraction"] > 0.01:
        failures.append("height scan is saturated at the high clip")
    if report["scan_std"] < 1e-4:
        failures.append("height scan is constant across generated terrain")

    report["failures"] = failures
    report["ok"] = not failures
except Exception:
    report["error"] = traceback.format_exc()[-3000:]
    report["failures"] = failures
finally:
    if env is not None:
        env.close()
    with open(args_cli.report, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    simulation_app.close()

raise SystemExit(0 if report.get("ok") else 1)
