"""Verify the V10.2 environment contract inside the simulator before training.

This probe exists because V10.1 ran for three thousand iterations with its duty
floor silently masked off. The term was registered at -3.0 and returned exactly
zero at every logged iteration, and no gate could see it: a reward term that is
masked to zero does not report itself as absent, it reports as satisfied.

``probe_v8_1.py`` already contained the check that catches this -- it drives the
term's internal duty estimate to an abandoned-foot value and requires the
penalty to fire. When the probe was ported to the new robot that block was
dropped; ``probe_v10_flat.py`` and ``probe_v10_1_flat.py`` keep only the line
that looks the term up, never use it, and still carry a docstring claiming the
check is present. It is restored below, and it fails on V10.1's configuration.

Two further checks are added that the V8.1 probe had no reason to contain:

* the moving mask is evaluated directly against the configured command band, so
  a future version that moves the command to another axis cannot disable the
  term by accident;
* the fore-aft axis check from V10.1, which caught the crab walk.
"""

import argparse
import json
import traceback

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-Velocity-Flat-Guide-RobotV2-Probe-v10-2-flat"
parser = argparse.ArgumentParser(description="Probe the V10.2 live-duty-floor environment.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/probe_v10_2_flat.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import v5_12_mdp

from v8_env_cfg import V8_REWARD_WEIGHTS
from v8_1_env_cfg import DUTY_CONTACT_FLOOR, DUTY_FLOOR_WEIGHT
from v10_2_flat_env_cfg import (
    FORWARD_AXIS,
    FORWARD_SIGN,
    GuideV2FlatDutyLiveEnvCfg,
    V10_2_FLAT_REWARD_WEIGHTS,
)
from v10_2_flat_ppo_cfg import V10_2_FLAT_PPO_CFG

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2FlatDutyLiveEnvCfg,
        "rsl_rl_cfg_entry_point": V10_2_FLAT_PPO_CFG,
    },
)

report = {"ok": False}
failures = []
try:
    cfg = GuideV2FlatDutyLiveEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    active = sorted(base.reward_manager.active_terms)
    report["active_terms"] = active
    if active != sorted(V10_2_FLAT_REWARD_WEIGHTS):
        failures.append(f"reward terms differ: {active}")

    weights = {n: float(base.reward_manager.get_term_cfg(n).weight) for n in active}
    report["weights"] = weights

    drifted = {n: weights[n] for n in V8_REWARD_WEIGHTS if weights.get(n) != V8_REWARD_WEIGHTS[n]}
    if drifted:
        failures.append(f"V8 weights drifted: {drifted}")
    if weights.get("duty_floor") != DUTY_FLOOR_WEIGHT:
        failures.append(f"duty_floor weight {weights.get('duty_floor')} != {DUTY_FLOOR_WEIGHT}")
    if abs(weights["duty_floor"]) >= weights["gait"]:
        failures.append("duty floor must stay weaker than the gait reward")

    obs = base.observation_manager.compute()
    report["obs_width"] = int(obs["policy"].shape[-1])
    if report["obs_width"] != 48:
        failures.append(f"obs width {report['obs_width']} != 48, V8 evaluator no longer applies")
    report["action_scale"] = float(base.cfg.actions.joint_pos.scale)
    if report["action_scale"] != 0.2:
        failures.append(f"action scale {report['action_scale']} != 0.2")

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
        failures.append(f"unexpected mass {report['mass_one_env_kg']}, expected ~45.45")

    termination_terms = list(base.termination_manager.active_terms)
    report["termination_terms"] = termination_terms
    if "base_tilt" not in termination_terms:
        failures.append(f"orientation termination missing: {termination_terms}")
    if "body_contact" in termination_terms:
        failures.append("contact termination must be off; base and hips own no collider")

    contact_sensor = base.scene.sensors["contact_forces"]
    report["contact_bodies"] = list(contact_sensor.body_names)
    if sorted(report["contact_bodies"]) != sorted(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
        failures.append(f"contact sensor resolved {report['contact_bodies']}")

    # -- THE V10.2 check: is the duty floor actually live under this command? ---
    # Direct first. The mask is what V10.1 lost, so read it rather than infer it.
    duty_col = base.reward_manager.active_terms.index("duty_floor")
    term = base.reward_manager.get_term_cfg("duty_floor").func
    term_params = base.reward_manager.get_term_cfg("duty_floor").params
    moving_threshold = float(term_params["moving_threshold"])
    mask = v5_12_mdp._moving_mask(base, term_params["command_name"], moving_threshold)
    report["moving_mask_fraction"] = round(float(mask.float().mean()), 4)
    # The command band is 0.3-0.9 on the forward axis and 0 lateral, so every
    # sampled command clears a 0.05 threshold except the standing-still ones the
    # command term deliberately injects.
    if report["moving_mask_fraction"] < 0.5:
        failures.append(
            f"moving mask is True for only {report['moving_mask_fraction']:.2f} of envs; "
            "the duty floor is masked off, which is the V10.1 defect"
        )

    action = torch.zeros((args_cli.num_envs, 12), device=base.device)
    for _ in range(40):
        env.step(action)

    step_reward = base.reward_manager._step_reward
    report["all_finite"] = bool(torch.isfinite(step_reward).all().item())
    if not report["all_finite"]:
        failures.append("non-finite reward")

    report["measured_duty_after_settling"] = [float(v) for v in term._duty.mean(dim=0)]

    # Loading all four feet must cost nothing -- the floor penalises abandonment,
    # not walking. Drive the duty estimate rather than relying on the robot to
    # stand: 40 steps after a randomized reset it is still settling, so a
    # measured stance would test the reset transient, not the term's contract.
    term._duty[:] = 1.0
    env.step(action)
    planted = float(base.reward_manager._step_reward[:, duty_col].mean())
    report["duty_penalty_when_planted"] = planted
    if planted < -1e-6:
        failures.append(f"duty floor penalizes fully loaded feet: {planted}")

    # Now abandon a foot and require the term to react. This is the assertion
    # V10.1 would have failed: with the command on +y and the mask reading the x
    # column, every one of these stays at exactly zero.
    term._duty[:] = 1.0
    term._duty[:, 0] = 0.0
    env.step(action)
    fired = base.reward_manager._step_reward[:, duty_col]
    report["duty_penalty_when_abandoned_mean"] = float(fired.mean())
    report["duty_penalty_when_abandoned_min"] = float(fired.min())
    fraction_firing = float((fired < -1e-3).float().mean())
    report["fraction_firing"] = fraction_firing
    if fraction_firing < 0.5:
        failures.append(f"duty floor did not fire on an abandoned foot ({fraction_firing:.2f})")
    if report["duty_penalty_when_abandoned_min"] >= -1e-3:
        failures.append("duty floor produced no penalty on any environment")

    report["contact_floor"] = DUTY_CONTACT_FLOOR

    # -- standing on its feet, measured frame-independently ---------------------
    env.reset()
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

    # -- the fore-aft axis, the check that caught the crab walk ------------------
    from isaaclab.utils.math import quat_apply_inverse

    names = list(robot.body_names)
    rp, rq = robot.data.root_pos_w[0], robot.data.root_quat_w[0]
    local = quat_apply_inverse(
        rq.unsqueeze(0).expand(len(names), -1), robot.data.body_pos_w[0] - rp
    )

    def _body(n):
        return local[names.index(n)]

    front = (_body("FL_foot") + _body("FR_foot")) / 2.0
    rear = (_body("RL_foot") + _body("RR_foot")) / 2.0
    separation = front - rear
    axis_index = int(separation[:2].abs().argmax())
    report["fore_aft_axis_measured"] = ["x", "y"][axis_index]
    report["fore_aft_separation_m"] = round(float(separation[axis_index]), 4)
    report["front_minus_rear"] = [round(float(v), 4) for v in separation]
    if report["fore_aft_axis_measured"] != FORWARD_AXIS:
        failures.append(
            f"fore-aft axis measures {report['fore_aft_axis_measured']}, config says {FORWARD_AXIS}"
        )
    measured_sign = 1 if report["fore_aft_separation_m"] > 0 else -1
    if measured_sign != FORWARD_SIGN:
        failures.append(
            f"head points {'+' if measured_sign > 0 else '-'}{FORWARD_AXIS}, config says "
            f"{'+' if FORWARD_SIGN > 0 else '-'}{FORWARD_AXIS}"
        )

    ranges = base.cfg.commands.base_velocity.ranges
    report["cmd_lin_vel_x"] = list(ranges.lin_vel_x)
    report["cmd_lin_vel_y"] = list(ranges.lin_vel_y)
    forward_range = ranges.lin_vel_y if FORWARD_AXIS == "y" else ranges.lin_vel_x
    lateral_range = ranges.lin_vel_x if FORWARD_AXIS == "y" else ranges.lin_vel_y
    if max(abs(v) for v in lateral_range) > 1e-6:
        failures.append(f"lateral command band is {lateral_range}, expected zero")
    if min(forward_range) * FORWARD_SIGN <= 0.0:
        failures.append(
            f"forward command band {forward_range} does not point along "
            f"{'+' if FORWARD_SIGN > 0 else '-'}{FORWARD_AXIS}"
        )
    # The commanded speed must clear the moving threshold, or the duty floor is
    # exempt for the entire run no matter which axis carries it.
    if min(abs(v) for v in forward_range) <= moving_threshold:
        failures.append(
            f"forward command band {forward_range} reaches the moving threshold "
            f"{moving_threshold}; the duty floor would be exempt at the low end"
        )

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
