"""Fine-tune the V11 actor through the frozen V12 depth front-end for torque use."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Guide-RobotV2-v13-depth-torque"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--actor-checkpoint", required=True)
parser.add_argument("--student-checkpoint", required=True)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initial-terrain-level", type=int, default=0)
parser.add_argument("--output-root", default="logs/rsl_rl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner

from training_resume import validate_initial_terrain_level
from v13_depth_policy import make_v13_depth_wrapper
from v13_torque_env_cfg import GuideV13TorqueEnvCfg, V13_REWARD_WEIGHTS
from v13_torque_ppo_cfg import V13_TORQUE_PPO_CFG


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV13TorqueEnvCfg,
        "rsl_rl_cfg_entry_point": V13_TORQUE_PPO_CFG,
    },
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path in (args_cli.actor_checkpoint, args_cli.student_checkpoint, args_cli.usd_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if args_cli.num_envs <= 0 or args_cli.max_iterations <= 0:
        raise ValueError("num_envs and max_iterations must be positive")

    env_cfg = GuideV13TorqueEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
    # V12 student data collection and closed-loop evaluation use clean proprioception.
    env_cfg.observations.policy.enable_corruption = False
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise ValueError("V13 requires V12's generated terrain")
    level = validate_initial_terrain_level(
        args_cli.initial_terrain_level, int(generator.num_rows)
    )
    env_cfg.scene.terrain.max_init_terrain_level = level
    env_cfg.seed = args_cli.seed

    agent_cfg = copy.deepcopy(V13_TORQUE_PPO_CFG)
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.max_iterations = args_cli.max_iterations
    log_root = Path(args_cli.output_root).resolve() / agent_cfg.experiment_name
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{agent_cfg.run_name}"
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] V13 log directory: {log_dir}", flush=True)

    manifest = {
        "actor_checkpoint": str(Path(args_cli.actor_checkpoint).resolve()),
        "actor_sha256": file_sha256(args_cli.actor_checkpoint),
        "frozen_v12_student": str(Path(args_cli.student_checkpoint).resolve()),
        "student_sha256": file_sha256(args_cli.student_checkpoint),
        "usd_path": str(Path(args_cli.usd_path).resolve()),
        "usd_sha256": file_sha256(args_cli.usd_path),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "initial_terrain_level": level,
        "reward_weights": V13_REWARD_WEIGHTS,
        "action_lpf_hz": 25.0,
        "control_dt_s": env_cfg.decimation * env_cfg.sim.dt,
        "filter_dt_s": env_cfg.sim.dt,
    }
    (log_dir / "v13_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("[INFO] Constructing V13 IsaacLab environment", flush=True)
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    print("[INFO] Environment constructed", flush=True)
    DepthWrapper = make_v13_depth_wrapper()
    env = DepthWrapper(
        raw_env,
        student_checkpoint=args_cli.student_checkpoint,
        clip_actions=agent_cfg.clip_actions,
    )
    print("[INFO] Frozen V12 depth wrapper constructed", flush=True)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        print("[INFO] PPO runner constructed", flush=True)
        runner.load(args_cli.actor_checkpoint, load_optimizer=False, map_location=agent_cfg.device)
        print("[INFO] V11 actor checkpoint loaded", flush=True)
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        print("[INFO] PPO iterations finished", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    failed = False
    try:
        main()
    except BaseException:
        traceback.print_exc()
        failed = True
    try:
        simulation_app.close()
    except BaseException:
        traceback.print_exc()
        failed = True
    if failed:
        raise SystemExit(1)
