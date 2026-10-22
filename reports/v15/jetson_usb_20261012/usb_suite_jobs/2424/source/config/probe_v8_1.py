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

TASK_ID = "Isaac-Velocity-Flat-Guide-Duty-Floor-Probe-v8-1"
parser = argparse.ArgumentParser(description="Probe the V8.1 duty-floor environment.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/probe_v8_1.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from v8_env_cfg import V8_REWARD_WEIGHTS
from v8_1_env_cfg import (
    DUTY_CONTACT_FLOOR,
    DUTY_FLOOR_WEIGHT,
    GuideDutyFloorEnvCfg,
    V8_1_REWARD_WEIGHTS,
)
from v8_1_ppo_cfg import V8_1_PPO_CFG

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": GuideDutyFloorEnvCfg, "rsl_rl_cfg_entry_point": V8_1_PPO_CFG},
)

report = {"ok": False}
failures = []
try:
    cfg = GuideDutyFloorEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped

    active = sorted(base.reward_manager.active_terms)
    report["active_terms"] = active
    if active != sorted(V8_1_REWARD_WEIGHTS):
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

    action = torch.zeros((args_cli.num_envs, 12), device=base.device)
    for _ in range(40):
        env.step(action)

    step_reward = base.reward_manager._step_reward
    report["all_finite"] = bool(torch.isfinite(step_reward).all().item())
    if not report["all_finite"]:
        failures.append("non-finite reward")

    report["measured_duty_after_settling"] = [float(v) for v in term._duty.mean(dim=0)]

    # Loading all four feet must cost nothing -- the floor is a penalty on
    # abandonment, not a tax on walking. Drive the duty estimate rather than
    # relying on the robot to stand still: 40 steps after a randomized reset the
    # robot is still settling, so a measured stance would test the reset
    # transient instead of the term's contract.
    term._duty[:] = 1.0
    env.step(action)
    planted = float(base.reward_manager._step_reward[:, duty_col].mean())
    report["duty_penalty_when_planted"] = planted
    if planted < -1e-6:
        failures.append(f"duty floor penalizes fully loaded feet: {planted}")

    # Now abandon a foot and require the term to react.
    term._duty[:] = 1.0
    term._duty[:, 0] = 0.0
    env.step(action)
    fired = base.reward_manager._step_reward[:, duty_col]
    report["duty_penalty_when_abandoned_mean"] = float(fired.mean())
    report["duty_penalty_when_abandoned_min"] = float(fired.min())
    # 90% of envs carry a move command; the rest are commanded to stand and are
    # correctly exempt, so require a clear majority to fire rather than all.
    fraction_firing = float((fired < -1e-3).float().mean())
    report["fraction_firing"] = fraction_firing
    if fraction_firing < 0.5:
        failures.append(f"duty floor did not fire on an abandoned foot ({fraction_firing:.2f})")
    if report["duty_penalty_when_abandoned_min"] >= -1e-3:
        failures.append("duty floor produced no penalty on any environment")

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
