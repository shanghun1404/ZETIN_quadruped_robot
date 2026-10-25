"""Train the V15-R blind fall-recovery policy from scratch on the V15 plant."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Recovery-Guide-RobotV2-v15"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd-path", required=True, help="V15 recovery USD with trunk colliders")
parser.add_argument("--resume-checkpoint", help="Continue a V15-R run: restore optimizer and iteration")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=48)
parser.add_argument("--output-root", default="logs/rsl_rl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from v15_env_cfg import V15_GAIN_SCALE_RANGE
from v15_math import make_finite_rollout
from v15_recovery_env_cfg import (
    DROP_ROOT_HEIGHT_M, FALLEN_FRACTION, RECOVERY_ACTION_SCALE, GuideV15RecoveryEnvCfg,
)
from v15_recovery_ppo_cfg import V15_RECOVERY_PPO_CFG


gym.register(id=TASK_ID, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)


FiniteRolloutWrapper = make_finite_rollout(RslRlVecEnvWrapper)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not os.path.isfile(args_cli.usd_path):
        raise FileNotFoundError(args_cli.usd_path)
    if args_cli.num_envs <= 0 or args_cli.max_iterations <= 0:
        raise ValueError("num_envs and max_iterations must be positive")
    env_cfg = GuideV15RecoveryEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
    env_cfg.seed = args_cli.seed
    agent_cfg = copy.deepcopy(V15_RECOVERY_PPO_CFG)
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.max_iterations = args_cli.max_iterations
    log_dir = (Path(args_cli.output_root).resolve() / agent_cfg.experiment_name
               / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"))
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] V15-R log directory: {log_dir}", flush=True)
    manifest = {
        "schema": 1,
        "usd_path": str(Path(args_cli.usd_path).resolve()),
        "usd_sha256": file_sha256(args_cli.usd_path),
        "resume_checkpoint": args_cli.resume_checkpoint,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "policy_observation": ["base_ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel_rel", "last_action"],
        "critic_extra_observation": ["base_lin_vel", "root_height"],
        "action_scale_rad": RECOVERY_ACTION_SCALE,
        "joint_drive": "physx_implicit_pd",
        "pd_gain_scale_range": list(V15_GAIN_SCALE_RANGE),
        "max_action_delay_steps": env_cfg.actions.joint_pos.max_delay_steps,
        "action_lpf_hz": 25.0,
        "effort_guard_fraction": env_cfg.actions.joint_pos.effort_guard_fraction,
        "fallen_fraction": FALLEN_FRACTION,
        "drop_root_height_m": DROP_ROOT_HEIGHT_M,
        "episode_length_s": env_cfg.episode_length_s,
        "reward_weights": {name: term.weight for name, term in vars(env_cfg.rewards).items()
                           if hasattr(term, "weight")},
        "control_dt_s": env_cfg.decimation * env_cfg.sim.dt,
        "physics_dt_s": env_cfg.sim.dt,
    }
    (log_dir / "v15_recovery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    env = FiniteRolloutWrapper(gym.make(TASK_ID, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    obs = env.get_observations()
    report = {key: (tuple(value.shape), int((~torch.isfinite(value)).sum())) for key, value in obs.items()}
    if obs["policy"].shape != (args_cli.num_envs, 42) or any(bad for _, bad in report.values()):
        raise RuntimeError(f"V15-R requires finite 42-dimensional proprioceptive observations: {report}")
    print(f"[INFO] V15-R obs policy={tuple(obs['policy'].shape)} privileged={tuple(obs['privileged'].shape)}", flush=True)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        if args_cli.resume_checkpoint:
            runner.load(args_cli.resume_checkpoint, load_optimizer=True, map_location=agent_cfg.device)
            runner.current_learning_iteration += 1
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print("[INFO] V15-R PPO iterations finished", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # Kit's shutdown hung with the GPU held after failures (jobs 258, 301, 315): exit now.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
