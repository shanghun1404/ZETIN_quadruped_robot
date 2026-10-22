"""Verify the V9 environment contract inside the simulator before training."""

import argparse
import json
import traceback

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-Velocity-Rough-Guide-Terrain-Probe-v9"
parser = argparse.ArgumentParser(description="Probe the V9 terrain environment.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/probe_v9.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from v9_env_cfg import GuideTerrainEnvCfg, V9_REWARD_WEIGHTS, SCAN_POINTS
from v8_env_cfg import TROT_PAIRS
from v9_ppo_cfg import V9_PPO_CFG

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": GuideTerrainEnvCfg, "rsl_rl_cfg_entry_point": V9_PPO_CFG},
)

report = {"ok": False}
failures = []
try:
    cfg = GuideTerrainEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    active = sorted(base.reward_manager.active_terms)
    report["active_terms"] = active
    if active != sorted(V9_REWARD_WEIGHTS):
        failures.append(f"reward terms differ: {active}")
    # The gait reward must be the dominant term -- that is the whole point.
    weights = {n: float(base.reward_manager.get_term_cfg(n).weight) for n in active}
    report["weights"] = weights
    if weights.get("gait") != 10.0:
        failures.append(f"gait weight is {weights.get('gait')}, expected 10.0")
    if weights["gait"] <= weights["base_linear_velocity"]:
        failures.append("gait reward must outweigh velocity tracking")

    report["action_scale"] = float(base.cfg.actions.joint_pos.scale)
    if report["action_scale"] != 0.2:
        failures.append(f"action scale {report['action_scale']} != 0.2")
    report["trot_pairs"] = [list(p) for p in TROT_PAIRS]
    report["track_air_time"] = bool(base.scene.sensors["contact_forces"].cfg.track_air_time)
    if not report["track_air_time"]:
        failures.append("contact sensor must track air time for the gait reward")
    report["decimation"] = int(base.cfg.decimation)
    report["sim_dt"] = float(base.cfg.sim.dt)
    report["control_hz"] = 1.0 / (base.cfg.sim.dt * base.cfg.decimation)

    obs = base.observation_manager.compute()
    report["obs_width"] = int(obs["policy"].shape[-1])
    # V8 observed 48 proprioceptive values; V9 adds the 187-point scan.
    report["expected_obs_width"] = 48 + SCAN_POINTS
    if report["obs_width"] != report["expected_obs_width"]:
        failures.append(f"obs width {report['obs_width']} != {report['expected_obs_width']}")

    scanner = base.scene.sensors.get("height_scanner")
    if scanner is None:
        failures.append("height_scanner missing -- V9 must sense terrain")
    else:
        hits = scanner.data.ray_hits_w
        report["scan_rays"] = int(hits.shape[1])
        if report["scan_rays"] != SCAN_POINTS:
            failures.append(f"scan has {report['scan_rays']} rays, expected {SCAN_POINTS}")
        report["scan_offset"] = list(base.cfg.scene.height_scanner.offset.pos)
        # Forward in the guide frame is base -x, so the scan must look that way.
        if report["scan_offset"][0] >= 0.0:
            failures.append(f"scanner offset x={report['scan_offset'][0]} does not look forward (-x)")

    report["terrain_type"] = str(base.cfg.scene.terrain.terrain_type)
    if report["terrain_type"] != "generator":
        failures.append(f"terrain_type {report['terrain_type']} != generator")
    report["max_init_terrain_level"] = int(base.cfg.scene.terrain.max_init_terrain_level)
    report["curriculum_terms"] = list(base.curriculum_manager.active_terms)
    if "terrain_levels" not in report["curriculum_terms"]:
        failures.append("terrain_levels curriculum missing")

    action = torch.zeros((args_cli.num_envs, 12), device=base.device)
    for _ in range(40):
        env.step(action)
    step_reward = base.reward_manager._step_reward
    report["all_finite"] = bool(torch.isfinite(step_reward).all().item())
    if not report["all_finite"]:
        failures.append("non-finite reward")
    idx = {n: i for i, n in enumerate(base.reward_manager.active_terms)}
    for term in ("gait", "air_time", "base_linear_velocity"):
        col = step_reward[:, idx[term]]
        report[f"{term}_mean"] = float(col.mean())
    # Standing at default pose must not be free under the gait objective.
    report["height_at_rest"] = float(base.scene["robot"].data.root_pos_w[:, 2].mean())
    scan_obs = base.observation_manager.compute()["policy"][:, 48:]
    # On generated terrain the scan must not be a constant vector, otherwise the
    # sensor is effectively absent and V9 reduces to V8 with dead inputs.
    report["scan_std_across_envs"] = float(scan_obs.std())
    if report["scan_std_across_envs"] < 1e-4:
        failures.append("height scan is constant -- terrain is flat or sensor is dead")
    env.close()
    report["failures"] = failures
    report["ok"] = not failures
except Exception:
    report["error"] = traceback.format_exc()[-1500:]
    report["failures"] = failures

with open(args_cli.report, "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
simulation_app.close()
raise SystemExit(0 if report.get("ok") else 1)
