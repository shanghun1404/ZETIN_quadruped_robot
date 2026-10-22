"""Verify the V9.2 environment contract inside the simulator before training.

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

TASK_ID = "Isaac-Velocity-Rough-Guide-Terrain-Probe-v9-2"
parser = argparse.ArgumentParser(description="Probe the V9.2 terrain environment.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/probe_v9_2.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from v8_env_cfg import V8_GUIDE_FORWARD_SPEED_RANGE
from v9_env_cfg import SCAN_POINTS
from v9_2_env_cfg import (
    EPISODE_LENGTH_S,
    GuideTerrainTurnBandEnvCfg,
    PROMOTION_DISTANCE_M,
    V9_2_REWARD_WEIGHTS,
    V9_2_YAW_RATE_RANGE,
    displacement_chord_m,
)
from v9_2_ppo_cfg import V9_2_PPO_CFG

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": GuideTerrainTurnBandEnvCfg, "rsl_rl_cfg_entry_point": V9_2_PPO_CFG},
)

report = {"ok": False}
failures = []
try:
    cfg = GuideTerrainTurnBandEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    # -- inherited from V8.1 / V9 / V9.1, all of which must survive -----------
    active = sorted(base.reward_manager.active_terms)
    weights = {n: float(base.reward_manager.get_term_cfg(n).weight) for n in active}
    report["weights"] = weights
    if active != sorted(V9_2_REWARD_WEIGHTS):
        failures.append(f"reward terms differ: {active}")
    if weights.get("duty_floor") != -3.0:
        failures.append(f"V8.1 duty floor lost: {weights.get('duty_floor')}")
    if weights.get("gait") != 10.0:
        failures.append(f"gait weight drifted: {weights.get('gait')}")

    term_cfg = base.termination_manager.get_term_cfg("body_contact")
    sensor_cfg = term_cfg.params["sensor_cfg"]
    sensor = base.scene.sensors[sensor_cfg.name]
    body_names = list(sensor.body_names)
    terminating = sorted(body_names[i] for i in sensor_cfg.body_ids)
    report["terminating_bodies"] = terminating
    if any("thigh" in n or "calf" in n for n in terminating):
        failures.append(f"V9.1 termination fix lost: {terminating}")
    if not any("base_link" in n for n in terminating):
        failures.append("base_link must still terminate")

    report["curriculum_terms"] = list(base.curriculum_manager.active_terms)
    if "terrain_levels" not in report["curriculum_terms"]:
        failures.append("terrain_levels curriculum missing")
    obs = base.observation_manager.compute()
    report["obs_width"] = int(obs["policy"].shape[-1])
    if report["obs_width"] != 48 + SCAN_POINTS:
        failures.append(f"obs width {report['obs_width']} != {48 + SCAN_POINTS}")

    # -- the isolated change, read back out of the live command manager -------
    command_term = base.command_manager.get_term("base_velocity")
    resolved_yaw = tuple(float(v) for v in command_term.cfg.ranges.ang_vel_z)
    report["yaw_range"] = list(resolved_yaw)
    if resolved_yaw != tuple(V9_2_YAW_RATE_RANGE):
        failures.append(f"yaw range {resolved_yaw} != {V9_2_YAW_RATE_RANGE}")
    report["heading_command"] = bool(command_term.cfg.heading_command)
    if report["heading_command"]:
        failures.append("heading_command must stay off -- the policy has to take yaw commands")

    # Sampled commands must actually respect the band, not just the config.
    sampled = command_term.command
    report["sampled_yaw_abs_max"] = float(sampled[:, 2].abs().max())
    report["sampled_speed_abs_max"] = float(sampled[:, 0].abs().max())
    if report["sampled_yaw_abs_max"] > max(abs(v) for v in resolved_yaw) + 1e-6:
        failures.append("sampled yaw exceeds the configured band")

    # -- the promotion arithmetic, on the resolved numbers --------------------
    slow, fast = V8_GUIDE_FORWARD_SPEED_RANGE
    worst_yaw = max(abs(v) for v in resolved_yaw)
    report["promotion_distance_m"] = PROMOTION_DISTANCE_M
    report["chord_fast_worst_yaw_m"] = displacement_chord_m(fast, worst_yaw, EPISODE_LENGTH_S)
    report["chord_slow_worst_yaw_m"] = displacement_chord_m(slow, worst_yaw, EPISODE_LENGTH_S)
    report["chord_fast_straight_m"] = displacement_chord_m(fast, 0.0, EPISODE_LENGTH_S)
    report["chord_slow_straight_m"] = displacement_chord_m(slow, 0.0, EPISODE_LENGTH_S)

    # The point of the change: at full commanded speed, even a sustained
    # worst-case turn must clear the promotion bar. If this fails the band is
    # still too wide and the curriculum will stall exactly as it did in V9.1.
    if report["chord_fast_worst_yaw_m"] <= PROMOTION_DISTANCE_M:
        failures.append(
            f"even at max speed a sustained {worst_yaw} rad/s turn only reaches "
            f"{report['chord_fast_worst_yaw_m']:.2f} m < {PROMOTION_DISTANCE_M} m"
        )
    # And walking straight must clear it at *any* commanded speed, or the
    # curriculum cannot promote even in the easiest case.
    if report["chord_slow_straight_m"] <= PROMOTION_DISTANCE_M:
        failures.append("straight-line travel at the slowest command cannot promote")

    action = torch.zeros((args_cli.num_envs, 12), device=base.device)
    for _ in range(20):
        env.step(action)
    report["all_finite"] = bool(torch.isfinite(base.reward_manager._step_reward).all().item())
    if not report["all_finite"]:
        failures.append("non-finite reward")

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
