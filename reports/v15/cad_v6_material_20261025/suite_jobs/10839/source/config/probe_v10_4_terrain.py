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

TASK_ID = "Isaac-Velocity-Rough-Guide-RobotV2-Probe-v10-4-terrain"
parser = argparse.ArgumentParser(description="Probe the V10.4 blind rough-terrain environment.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/probe_v10_4_terrain.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import v5_12_mdp

from v8_env_cfg import V8_REWARD_WEIGHTS
from v8_1_env_cfg import DUTY_CONTACT_FLOOR, DUTY_FLOOR_WEIGHT
from v10_4_terrain_env_cfg import (
    FORWARD_AXIS,
    FORWARD_SIGN,
    GuideV2BlindTerrainEnvCfg,
    PROMOTION_DISTANCE_M,
    TERRAIN_SPAWN_ROOT_FRAME_Z_M,
    V10_4_REWARD_WEIGHTS,
    V10_4_YAW_RATE_RANGE,
    displacement_chord_m,
)
from v10_4_terrain_ppo_cfg import V10_4_TERRAIN_PPO_CFG
from guide_robot_v2_cfg import FEET_BELOW_ROOT_FRAME_M, FEET_SPAWN_CLEARANCE_M

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2BlindTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": V10_4_TERRAIN_PPO_CFG,
    },
)

report = {"ok": False}
failures = []
try:
    cfg = GuideV2BlindTerrainEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    # -- THE V10.3 check: does the robot start above the floor? -----------------
    # Every earlier run on this robot spawned 0.81 m underground and was ejected
    # by the solver. The trace: root z -0.3700 before any step, ground support
    # 2.432x body weight on step 1, rising 0.815 m over ~200 steps. Both numbers
    # are asserted here, because the settled state looked perfectly healthy and
    # is what every previous check measured.
    robot_pre = base.scene["robot"]
    sensor_pre = base.scene.sensors["contact_forces"]
    weight_pre = float(robot_pre.root_physx_view.get_masses()[0].sum()) * 9.81

    report["configured_spawn_z"] = round(float(base.cfg.scene.robot.init_state.pos[2]), 4)
    # On terrain the robot spawns at env_origin + init_state.pos, and each tile's
    # origin sits at its own local surface, so the world z is meaningless on its
    # own. Measure the height ABOVE THE TILE ORIGIN, which is what the spawn
    # constant actually controls.
    origins_z = base.scene.env_origins[:, 2]
    rel_z = robot_pre.data.root_pos_w[:, 2] - origins_z
    report["env_origin_z_mean"] = round(float(origins_z.mean()), 4)
    report["env_origin_z_spread"] = round(float(origins_z.max() - origins_z.min()), 4)
    report["root_z_before_any_step"] = round(float(rel_z.mean()), 4)
    report["root_z_before_any_step_min"] = round(float(rel_z.min()), 4)
    expected = FEET_BELOW_ROOT_FRAME_M + FEET_SPAWN_CLEARANCE_M
    report["expected_spawn_z"] = round(float(expected), 4)
    if report["root_z_before_any_step"] <= 0.0:
        failures.append(
            f"robot spawns at z={report['root_z_before_any_step']}, at or below the floor; "
            "this is the V10.1/V10.2 defect"
        )
    # Absolute, not relative: the quantity is a world-space offset in metres and a
    # relative bound would change meaning with the clearance. The settled
    # measurements span 0.4357-0.4455 (9.8 mm), so 15 mm covers the spread with
    # margin while a sign error -- off by 0.89 m -- cannot slip through.
    if abs(report["root_z_before_any_step"] - expected) > 0.015:
        failures.append(
            f"spawn z {report['root_z_before_any_step']} is not within 15 mm of the "
            f"expected {report['expected_spawn_z']} (= feet-below-root + clearance)"
        )
    if abs(report["configured_spawn_z"] - TERRAIN_SPAWN_ROOT_FRAME_Z_M) > 1e-6:
        failures.append("init_state.pos.z does not match SPAWN_ROOT_FRAME_Z_M")

    # First-step contact force. The defect registered 2.432x body weight. A robot
    # placed 5 cm above the floor falls 2 mm in one 20 ms step and touches nothing,
    # so the bar is 0.5 -- low enough to reject partial penetration, not just the
    # extreme case. An earlier version used 1.5, which would have passed a robot
    # meaningfully embedded in the ground while the comment claimed otherwise.
    _zero = torch.zeros((args_cli.num_envs, 12), device=base.device)
    env.step(_zero)
    first_force = sensor_pre.data.net_forces_w.norm(dim=-1).sum(dim=1)
    report["support_on_first_step"] = round(float(first_force.mean()) / weight_pre, 3)
    if report["support_on_first_step"] > 0.5:
        failures.append(
            f"ground support on the first step is {report['support_on_first_step']}x body "
            "weight -- the robot is starting inside the floor"
        )

    active = sorted(base.reward_manager.active_terms)
    report["active_terms"] = active
    if active != sorted(V10_4_REWARD_WEIGHTS):
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

    # -- terrain, curriculum and the blind contract -----------------------------
    terrain = base.scene.terrain
    report["terrain_type"] = str(base.cfg.scene.terrain.terrain_type)
    if report["terrain_type"] != "generator":
        failures.append(f"terrain_type is {report['terrain_type']}, expected 'generator'")
    report["max_init_terrain_level"] = int(base.cfg.scene.terrain.max_init_terrain_level)
    if report["max_init_terrain_level"] != 0:
        failures.append("every environment must start on the easiest tile")
    report["terrain_levels_at_reset"] = round(float(terrain.terrain_levels.float().mean()), 4)

    # THE V9 trap: V8's __post_init__ nulls the curriculum after the class
    # attribute binds, so a declared term can vanish without any error.
    curriculum_terms = list(base.curriculum_manager.active_terms)
    report["curriculum_terms"] = curriculum_terms
    if "terrain_levels" not in curriculum_terms:
        failures.append(
            f"terrain_levels curriculum missing ({curriculum_terms}); the run would sit "
            "on the easiest tile forever, which is exactly what happened to V9"
        )

    # Blind is the point of this version: no scanner, and the observation must
    # stay at V8's width so a V10.3 and a V10.4 checkpoint remain comparable.
    report["height_scanner_present"] = "height_scanner" in base.scene.sensors
    if report["height_scanner_present"]:
        failures.append("a height scanner is present; V10.4 is blind by design")
    if "height_scan" in base.observation_manager.active_terms["policy"]:
        failures.append("height_scan observation present on a blind configuration")

    # The chord arithmetic that makes the yaw band a precondition.
    ranges_c = base.cfg.commands.base_velocity.ranges
    report["cmd_ang_vel_z"] = list(ranges_c.ang_vel_z)
    if tuple(ranges_c.ang_vel_z) != tuple(V10_4_YAW_RATE_RANGE):
        failures.append(f"yaw band {ranges_c.ang_vel_z} != {V10_4_YAW_RATE_RANGE}")
    worst_yaw = max(abs(v) for v in ranges_c.ang_vel_z)
    chord = displacement_chord_m(0.5, worst_yaw, float(base.cfg.episode_length_s))
    report["worst_case_chord_m"] = round(chord, 3)
    report["promotion_distance_m"] = PROMOTION_DISTANCE_M
    if chord <= PROMOTION_DISTANCE_M:
        failures.append(
            f"at yaw {worst_yaw} the 20 s displacement chord is {chord:.3f} m against a "
            f"{PROMOTION_DISTANCE_M} m promotion threshold; no environment could ever "
            "be promoted and the run would be wasted"
        )
    tile = base.cfg.scene.terrain.terrain_generator.size[0]
    report["terrain_tile_size_m"] = float(tile)
    if abs(tile / 2.0 - PROMOTION_DISTANCE_M) > 1e-6:
        failures.append(
            f"tile size {tile} implies a promotion distance of {tile / 2.0}, but the "
            f"chord check used {PROMOTION_DISTANCE_M}"
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
