"""Verify the V8.1 environment contract inside the simulator before training.

Beyond V8's checks this probe proves the new term is *live*: a reward term that
is wired up but silent is exactly the failure V6 shipped, where ``base_height_l2``
was present, weighted, and produced 1.4 mm of movement. So the probe drives the
term's internal duty estimate to an abandoned-foot value and requires the penalty
to actually fire, rather than trusting that it would.
"""

import argparse
import json
import traceback

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-Velocity-Flat-Guide-RobotV2-Probe-v10-flat"
parser = argparse.ArgumentParser(description="Probe the V8.1 duty-floor environment.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/probe_v10_flat.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from v8_env_cfg import V8_REWARD_WEIGHTS
from v8_1_env_cfg import DUTY_CONTACT_FLOOR, DUTY_FLOOR_WEIGHT
from v10_flat_env_cfg import (
    GuideV2FlatEnvCfg,
    V10_FLAT_REWARD_WEIGHTS,
)
from v10_flat_ppo_cfg import V10_FLAT_PPO_CFG

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": GuideV2FlatEnvCfg, "rsl_rl_cfg_entry_point": V10_FLAT_PPO_CFG},
)

report = {"ok": False}
failures = []
try:
    cfg = GuideV2FlatEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    active = sorted(base.reward_manager.active_terms)
    report["active_terms"] = active
    if active != sorted(V10_FLAT_REWARD_WEIGHTS):
        failures.append(f"reward terms differ: {active}")

    weights = {n: float(base.reward_manager.get_term_cfg(n).weight) for n in active}
    report["weights"] = weights

    # V8.1 adds exactly one term; the other fourteen must be untouched.
    drifted = {n: weights[n] for n in V8_REWARD_WEIGHTS if weights.get(n) != V8_REWARD_WEIGHTS[n]}
    if drifted:
        failures.append(f"V8 weights drifted: {drifted}")
    if weights.get("duty_floor") != DUTY_FLOOR_WEIGHT:
        failures.append(f"duty_floor weight {weights.get('duty_floor')} != {DUTY_FLOOR_WEIGHT}")
    # The floor complements the gait reward; it must never be able to override it.
    if abs(weights["duty_floor"]) >= weights["gait"]:
        failures.append("duty floor must stay weaker than the gait reward")

    # Reusing evaluate_v8.py / play_v8_video.py on a V8.1 checkpoint is only
    # valid because the policy sees exactly what V8's policy saw.
    obs = base.observation_manager.compute()
    report["obs_width"] = int(obs["policy"].shape[-1])
    if report["obs_width"] != 48:
        failures.append(f"obs width {report['obs_width']} != 48, V8 evaluator no longer applies")
    report["action_scale"] = float(base.cfg.actions.joint_pos.scale)
    if report["action_scale"] != 0.2:
        failures.append(f"action scale {report['action_scale']} != 0.2")

    duty_col = base.reward_manager.active_terms.index("duty_floor")
    term = base.reward_manager.get_term_cfg("duty_floor").func

    # -- the robot, and the three things that broke on this asset today -------
    robot = base.scene["robot"]
    report["body_count"] = len(robot.body_names)
    report["root_body"] = robot.body_names[0]
    if report["root_body"] != "base":
        failures.append(f"root body is {report['root_body']}, expected 'base'")
    report["joint_count"] = len(robot.joint_names)
    if report["joint_count"] != 12:
        failures.append(f"expected 12 joints, found {report['joint_count']}")
    masses = robot.root_physx_view.get_masses()
    report["mass_one_env_kg"] = round(float(masses[0].sum()), 3)
    if not (43.0 < report["mass_one_env_kg"] < 50.0):
        failures.append(f"unexpected mass {report['mass_one_env_kg']}, expected ~45.45 +/- randomisation")

    active_terms = list(base.termination_manager.active_terms)
    report["termination_terms"] = active_terms
    if "base_tilt" not in active_terms:
        failures.append(f"orientation termination missing: {active_terms}")
    if "body_contact" in active_terms:
        failures.append("contact termination must be off; base and hips own no collider")

    contact_sensor = base.scene.sensors["contact_forces"]
    report["contact_bodies"] = list(contact_sensor.body_names)
    if sorted(report["contact_bodies"]) != sorted(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
        failures.append(f"contact sensor resolved {report['contact_bodies']}")

    # THE check today's failures needed. Body-frame positions are displaced from
    # the geometry on this asset, so "is it standing?" cannot be answered with
    # root_pos_w. Contact force and projected gravity are frame-independent.
    # Two runs were wasted before this existed: one where the robot fell through
    # the terrain (0 N of support, root z reaching -314 m) and one where it landed
    # and flipped to 166 degrees, both reported as healthy by every other check.
    settle = torch.zeros((args_cli.num_envs, 12), device=base.device)
    for _ in range(400):
        env.step(settle)
    force = contact_sensor.data.net_forces_w.norm(dim=-1).sum(dim=1)
    weight = float(masses[0].sum()) * 9.81
    report["support_fraction"] = round(float(force.mean()) / weight, 3)
    gravity = robot.data.projected_gravity_b
    tilt_deg = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
    report["tilt_deg_mean"] = round(float(tilt_deg.mean()), 3)
    report["tilt_deg_max"] = round(float(tilt_deg.max()), 3)
    report["settled_root_z"] = round(float(robot.data.root_pos_w[:, 2].mean()), 4)
    if not (0.80 < report["support_fraction"] < 1.25):
        failures.append(
            f"the ground carries {report['support_fraction']:.2f} of the robot's weight; "
            "it is not standing on its feet"
        )
    if report["tilt_deg_mean"] > 20.0:
        failures.append(f"robot settles tilted at {report['tilt_deg_mean']} degrees")

    report["contact_floor"] = DUTY_CONTACT_FLOOR
    report["height_at_rest"] = float(base.scene["robot"].data.root_pos_w[:, 2].mean())
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
