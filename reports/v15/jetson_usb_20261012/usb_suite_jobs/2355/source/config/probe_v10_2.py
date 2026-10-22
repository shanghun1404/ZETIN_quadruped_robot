"""Verify the V9.3 environment contract inside the simulator before training.

V9.2's claim is arithmetic about the curriculum: "with this yaw band, a robot
walking at commanded speed travels far enough to be promoted." So the probe
resolves the *actual* sampled command range out of the running environment and
runs the promotion arithmetic against it, rather than reading a tuple back out
of the config it just wrote.

It also re-checks the two properties V9.1 and V9 established, because both are
inherited through two levels now and inheritance is where they get lost.
"""

import argparse
import json
import traceback

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-Velocity-Rough-Guide-Terrain-Probe-v10-2"
parser = argparse.ArgumentParser(description="Probe the V9.4 terrain environment.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/probe_v10_2.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp

from v8_env_cfg import V8_GUIDE_FORWARD_SPEED_RANGE
from guide_frame import guide_forward_speed_range_to_base
from v9_env_cfg import SCAN_POINTS
from v9_2_env_cfg import (
    EPISODE_LENGTH_S,
    PROMOTION_DISTANCE_M,
    V9_2_YAW_RATE_RANGE,
    displacement_chord_m,
)
from v9_3_env_cfg import (
    OPERATING_SPEED_M_S,
    V9_2_LIN_VEL_STD,
    V9_3_LIN_VEL_STD,
    standing_still_reward_per_s,
)
from v9_4_env_cfg import GAIT_FULL_CREDIT_SPEED_M_S
from v10_2_env_cfg import (
    GuideV2TurnBandEnvCfg,
    V10_2_REWARD_WEIGHTS,
)
from v10_2_ppo_cfg import V10_2_PPO_CFG
import v9_4_mdp

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": GuideV2TurnBandEnvCfg, "rsl_rl_cfg_entry_point": V10_2_PPO_CFG},
)


# What this version is expected to have, which is not what V9.4 has.
EXPECTED_YAW_RANGE = (-0.2, 0.2)
EXPECTED_LIN_VEL_STD = 1.0
EXPECTED_GAIT_FUNC = "GaitReward"
CHORD_MUST_CLEAR = True

report = {"ok": False}
failures = []
env = None
try:
    cfg = GuideV2TurnBandEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    # -- inherited from V8.1 / V9 / V9.1, all of which must survive -----------
    active = sorted(base.reward_manager.active_terms)
    weights = {n: float(base.reward_manager.get_term_cfg(n).weight) for n in active}
    report["weights"] = weights
    if active != sorted(V10_2_REWARD_WEIGHTS):
        failures.append(f"reward terms differ: {active}")
    for name, expected in V10_2_REWARD_WEIGHTS.items():
        if name in weights and abs(weights[name] - expected) > 1.0e-9:
            failures.append(
                f"reward weight {name} drifted: {weights[name]} != {expected}"
            )
    if weights.get("duty_floor") != -3.0:
        failures.append(f"V8.1 duty floor lost: {weights.get('duty_floor')}")
    if weights.get("gait") != 10.0:
        failures.append(f"gait weight drifted: {weights.get('gait')}")

    # -- V10 termination: orientation, not contact ---------------------------
    # The MJCF gives collision geometry to the four feet only, so base and the
    # hips own no collider and illegal_contact could never fire. Measured:
    # adding bounding-box colliders to them dropped ground support from 99.4% to
    # 56% because the boxes struck the ground and lifted two feet.
    active_terms = list(base.termination_manager.active_terms)
    report["termination_terms"] = active_terms
    if "base_tilt" not in active_terms:
        failures.append(f"orientation termination missing: {active_terms}")
    if "body_contact" in active_terms:
        failures.append("contact termination must be off; those bodies own no collider")
    tilt_cfg = base.termination_manager.get_term_cfg("base_tilt")
    report["tilt_limit_rad"] = float(tilt_cfg.params["limit_angle"])
    if not (0.5 < report["tilt_limit_rad"] < 1.6):
        failures.append(f"implausible tilt limit {report['tilt_limit_rad']}")

    # -- the robot itself ----------------------------------------------------
    robot = base.scene["robot"]
    report["body_names_count"] = len(robot.body_names)
    # get_masses() is [num_envs, num_bodies]; summing everything counts all 64
    # robots, and add_base_mass has already randomised the trunk by (-2, +4).
    masses = robot.root_physx_view.get_masses()
    report["total_mass_kg"] = round(float(masses[0].sum()), 3)
    report["mass_spread_kg"] = round(float(masses.sum(dim=1).max() - masses.sum(dim=1).min()), 3)
    # 45.451 kg nominal, plus the event's (-2, +4) trunk randomisation.
    if not (43.0 < report["total_mass_kg"] < 50.0):
        failures.append(f"unexpected robot mass {report['total_mass_kg']}, expected ~45.45")
    report["root_body"] = robot.body_names[0]
    if report["root_body"] != "base":
        failures.append(f"root body is {report['root_body']}, expected 'base'")
    for needed in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        if needed not in robot.body_names:
            failures.append(f"missing body {needed}")
    report["joint_names"] = list(robot.joint_names)
    if len(report["joint_names"]) != 12:
        failures.append(f"expected 12 joints, found {len(report['joint_names'])}")

    report["curriculum_terms"] = list(base.curriculum_manager.active_terms)
    if "terrain_levels" not in report["curriculum_terms"]:
        failures.append("terrain_levels curriculum missing")
    obs = base.observation_manager.compute()
    report["obs_width"] = int(obs["policy"].shape[-1])
    if report["obs_width"] != 48 + SCAN_POINTS:
        failures.append(f"obs width {report['obs_width']} != {48 + SCAN_POINTS}")

    # -- V10.1/V10.2 contract: this is a *baseline*, not the V9.4 objective -----
    # V10.1 deliberately reverts V9.2's yaw band, V9.3's kernel width and V9.4's
    # speed ramp, so the probe must assert those reverted values, not the fixed
    # ones. Asserting the V9.4 contract here would fail a correct baseline.
    vel_cfg = base.reward_manager.get_term_cfg("base_linear_velocity")
    report["lin_vel_std"] = float(vel_cfg.params["std"])
    report["lin_vel_weight"] = float(vel_cfg.weight)
    if report["lin_vel_std"] != EXPECTED_LIN_VEL_STD:
        failures.append(f"lin vel std {report['lin_vel_std']} != {EXPECTED_LIN_VEL_STD}")
    if report["lin_vel_weight"] != 5.0:
        failures.append("velocity weight must stay at the published 5.0")

    gait_cfg = base.reward_manager.get_term_cfg("gait")
    report["gait_func"] = type(gait_cfg.func).__name__
    report["gait_weight"] = float(gait_cfg.weight)
    if report["gait_func"] != EXPECTED_GAIT_FUNC:
        failures.append(f"gait term is {report['gait_func']}, expected {EXPECTED_GAIT_FUNC}")
    if report["gait_weight"] != 10.0:
        failures.append("gait weight must stay at the published 10.0")

    command_term = base.command_manager.get_term("base_velocity")
    resolved_base_x = tuple(float(v) for v in command_term.cfg.ranges.lin_vel_x)
    expected_base_x = guide_forward_speed_range_to_base(V8_GUIDE_FORWARD_SPEED_RANGE)
    report["base_x_range"] = list(resolved_base_x)
    if resolved_base_x != expected_base_x:
        failures.append(f"base x range {resolved_base_x} != {expected_base_x}")
    resolved_yaw = tuple(float(v) for v in command_term.cfg.ranges.ang_vel_z)
    report["yaw_range"] = list(resolved_yaw)
    if resolved_yaw != EXPECTED_YAW_RANGE:
        failures.append(f"yaw range {resolved_yaw} != {EXPECTED_YAW_RANGE}")
    if bool(command_term.cfg.heading_command):
        failures.append("heading_command must stay off -- the policy has to take yaw commands")

    sampled = command_term.command
    moving_x = sampled[:, 0][sampled[:, 0].abs() > 1.0e-6]
    report["sampled_moving_count"] = int(moving_x.numel())
    if not moving_x.numel():
        failures.append("no moving command was sampled")
    elif bool((moving_x >= 0.0).any()):
        failures.append("guide-forward commands must be negative in base x")
    else:
        report["sampled_guide_speed_min"] = float((-moving_x).min())
        report["sampled_guide_speed_max"] = float((-moving_x).max())

    # The promotion chord is a *diagnostic* here, not a gate. At the baseline's
    # +/-0.6 band it is expected to fall short of the 4.0 m promotion distance --
    # that is the defect V10.2 exists to fix, reproduced on the new robot.
    slow, fast = V8_GUIDE_FORWARD_SPEED_RANGE
    worst_yaw = max(abs(v) for v in resolved_yaw)
    report["promotion_distance_m"] = PROMOTION_DISTANCE_M
    report["chord_fast_worst_yaw_m"] = displacement_chord_m(fast, worst_yaw, EPISODE_LENGTH_S)
    report["chord_slow_straight_m"] = displacement_chord_m(slow, 0.0, EPISODE_LENGTH_S)
    report["chord_clears_promotion"] = (
        report["chord_fast_worst_yaw_m"] > PROMOTION_DISTANCE_M
    )
    if CHORD_MUST_CLEAR and not report["chord_clears_promotion"]:
        failures.append(
            f"a sustained {worst_yaw} rad/s turn reaches only "
            f"{report['chord_fast_worst_yaw_m']:.2f} m < {PROMOTION_DISTANCE_M} m"
        )
    # Walking straight must promote at any commanded speed, in every version.
    if report["chord_slow_straight_m"] <= PROMOTION_DISTANCE_M:
        failures.append("straight-line travel at the slowest command cannot promote")


    action = torch.zeros((args_cli.num_envs, 12), device=base.device)
    for _ in range(20):
        env.step(action)
    report["all_finite"] = bool(torch.isfinite(base.reward_manager._step_reward).all().item())
    if not report["all_finite"]:
        failures.append("non-finite reward")

    # A stationary robot must not collect the gait reward. Zeroing root velocity
    # and stepping once is the closest in-sim analogue of the failure V9.4
    # targets, where V9.2/V9.3 collected ~6.7 per second without travelling.
    robot = base.scene["robot"]
    gait_col = base.reward_manager.active_terms.index("gait")
    moving_gait = float(base.reward_manager._step_reward[:, gait_col].mean())
    zero_vel = torch.zeros_like(robot.data.root_vel_w)
    robot.write_root_velocity_to_sim(zero_vel)
    env.step(action)
    stopped_gait = float(base.reward_manager._step_reward[:, gait_col].mean())
    report["gait_reward_while_moving"] = moving_gait
    report["gait_reward_after_zeroing_velocity"] = stopped_gait
    report["stationary_speed_m_s"] = float(
        torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1).mean()
    )
    if abs(stopped_gait) > 0.5 * GAIT_FULL_CREDIT_SPEED_M_S * 10.0:
        failures.append(
            f"gait reward {stopped_gait} survives a stationary robot -- the ramp is not applied"
        )

    report["failures"] = failures
    report["ok"] = not failures
except Exception:
    report["error"] = traceback.format_exc()[-1500:]
    report["failures"] = failures
finally:
    if env is not None:
        env.close()

with open(args_cli.report, "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
simulation_app.close()
raise SystemExit(0 if report.get("ok") else 1)
