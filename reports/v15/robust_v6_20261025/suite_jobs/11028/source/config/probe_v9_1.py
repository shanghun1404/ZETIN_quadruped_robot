"""Verify the V9.1 environment contract inside the simulator before training.

The whole point of V9.1 is a *behavioural* claim -- "a shin touching a stair no
longer ends the episode" -- so this probe drives contact forces into the sensor
buffer and reads the termination back, rather than checking that a regex string
looks right. A regex that resolves to the wrong body set would pass a source
inspection and still burn five hours.
"""

import argparse
import json
import traceback

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-Velocity-Rough-Guide-Terrain-Probe-v9-1"
parser = argparse.ArgumentParser(description="Probe the V9.1 terrain environment.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/probe_v9_1.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from v9_env_cfg import SCAN_POINTS
from v9_1_env_cfg import (
    GuideTerrainFixedEnvCfg,
    TERMINATION_BODIES,
    TERMINATION_THRESHOLD_N,
    TOLERATED_CONTACT_BODIES,
    V9_1_REWARD_WEIGHTS,
)
from v9_1_ppo_cfg import V9_1_PPO_CFG

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": GuideTerrainFixedEnvCfg, "rsl_rl_cfg_entry_point": V9_1_PPO_CFG},
)

report = {"ok": False}
failures = []
try:
    cfg = GuideTerrainFixedEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    # -- the reward table and the terrain setup must be exactly V9's ----------
    active = sorted(base.reward_manager.active_terms)
    report["active_terms"] = active
    if active != sorted(V9_1_REWARD_WEIGHTS):
        failures.append(f"reward terms differ from V9: {active}")
    weights = {n: float(base.reward_manager.get_term_cfg(n).weight) for n in active}
    report["weights"] = weights
    drifted = {n: weights[n] for n in V9_1_REWARD_WEIGHTS if weights.get(n) != V9_1_REWARD_WEIGHTS[n]}
    if drifted:
        failures.append(f"reward weights drifted: {drifted}")
    # V8.1's duty floor must survive two levels of inheritance.
    if weights.get("duty_floor") != -3.0:
        failures.append(f"duty_floor lost in inheritance: {weights.get('duty_floor')}")

    report["curriculum_terms"] = list(base.curriculum_manager.active_terms)
    if "terrain_levels" not in report["curriculum_terms"]:
        failures.append("terrain_levels curriculum missing")
    report["terrain_type"] = str(base.cfg.scene.terrain.terrain_type)
    obs = base.observation_manager.compute()
    report["obs_width"] = int(obs["policy"].shape[-1])
    report["expected_obs_width"] = 48 + SCAN_POINTS
    if report["obs_width"] != report["expected_obs_width"]:
        failures.append(f"obs width {report['obs_width']} != {report['expected_obs_width']}")

    # -- the isolated change, tested by behaviour -----------------------------
    term_cfg = base.termination_manager.get_term_cfg("body_contact")
    sensor_cfg = term_cfg.params["sensor_cfg"]
    sensor = base.scene.sensors[sensor_cfg.name]
    body_names = list(sensor.body_names)
    terminating = sorted(body_names[i] for i in sensor_cfg.body_ids)
    report["terminating_bodies"] = terminating
    report["termination_regex"] = TERMINATION_BODIES
    report["threshold_N"] = float(term_cfg.params["threshold"])

    # The two body classes that had to stop ending episodes.
    tolerated = sorted(n for n in body_names if "thigh" in n or "calf" in n)
    report["tolerated_bodies"] = tolerated
    report["tolerated_patterns"] = list(TOLERATED_CONTACT_BODIES)
    if not tolerated:
        failures.append("robot has no thigh/calf bodies -- probe cannot test the claim")
    if any(n in terminating for n in tolerated):
        failures.append(f"thigh/calf still terminate: {terminating}")
    if not any("base_link" in n for n in terminating):
        failures.append("base_link must still terminate")
    if not any("hip" in n for n in terminating):
        failures.append("hip must still terminate")

    forces = sensor.data.net_forces_w_history
    name_to_idx = {n: i for i, n in enumerate(body_names)}

    def fires_when_loaded(names, load=500.0):
        """Write a large contact force onto `names` and read the termination back."""
        forces.zero_()
        for n in names:
            forces[:, :, name_to_idx[n], 2] = load
        out = term_cfg.func(base, **term_cfg.params)
        return float(out.float().mean())

    report["fires_on_thigh_calf"] = fires_when_loaded(tolerated)
    report["fires_on_base_link"] = fires_when_loaded(["base_link"])
    hips = [n for n in body_names if "hip" in n]
    report["fires_on_hip"] = fires_when_loaded(hips[:1]) if hips else None
    report["fires_on_feet"] = fires_when_loaded([n for n in body_names if "foot" in n])
    forces.zero_()

    # A shin against a stair edge must be survivable; a body on the ground must not.
    if report["fires_on_thigh_calf"] != 0.0:
        failures.append("loading every thigh and calf still terminates the episode")
    if report["fires_on_base_link"] != 1.0:
        failures.append("base_link contact no longer terminates -- hardware protection lost")
    if report["fires_on_hip"] not in (None, 1.0):
        failures.append("hip contact no longer terminates")
    if report["fires_on_feet"] != 0.0:
        failures.append("foot contact terminates, which would end every episode")

    report["failures"] = failures
    report["ok"] = not failures
    env.close()
except Exception:
    report["error"] = traceback.format_exc()[-1500:]
    report["failures"] = failures

with open(args_cli.report, "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
simulation_app.close()
raise SystemExit(0 if report.get("ok") else 1)
