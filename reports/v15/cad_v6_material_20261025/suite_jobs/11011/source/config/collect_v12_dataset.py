"""Collect synchronized depth, proprioception, scan, and V11 action shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Velocity-Rough-Guide-RobotV2-v12-depth-collection"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--split", choices=("train", "validation", "dagger"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--samples", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--shard-size", type=int, default=2048)
parser.add_argument("--warmup-steps", type=int, default=50)
parser.add_argument("--sample-stride", type=int, default=4)
parser.add_argument("--rollout-policy", choices=("teacher", "student"), default="teacher")
parser.add_argument("--student-checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from evaluation_cli import load_inference_checkpoint
from v11_terrain_env_cfg import GuideV2ScannerTerrainEnvCfg
from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG
from v12_camera import camera_depth_nchw, camera_spec, make_v12_depth_camera_cfg
from v12_collection import (
    DatasetShardWriter,
    policy_observation_tensor,
    select_rollout_action,
)
from v12_teacher_actor import load_frozen_v11_actor
from v12_vision_student import TinyDepthViT, reconstruct_teacher_observation


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2ScannerTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": V11_TERRAIN_PPO_CFG,
    },
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_student(path: str | Path, device: str) -> TinyDepthViT:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("student checkpoint is missing model_state_dict")
    model = TinyDepthViT()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def pin_all_terrain_rows_and_types(raw_env) -> None:
    base_env = raw_env.unwrapped
    terrain = base_env.scene.terrain
    generator = base_env.cfg.scene.terrain.terrain_generator
    if generator is None:
        raise ValueError("V12 collection requires generated terrain")
    rows = int(generator.num_rows)
    columns = int(generator.num_cols)
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    terrain.terrain_levels[:] = env_ids % rows
    terrain.terrain_types[:] = (env_ids // rows) % columns
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels,
        terrain.terrain_types,
    ]
    raw_env.reset()


def main() -> dict[str, object]:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.rollout_policy == "student" and not args_cli.student_checkpoint:
        raise ValueError("--student-checkpoint is required for student rollout")
    if args_cli.samples <= 0 or args_cli.num_envs <= 0:
        raise ValueError("--samples and --num-envs must be positive")
    if args_cli.warmup_steps < 0 or args_cli.sample_stride <= 0:
        raise ValueError("warmup must be non-negative and stride must be positive")

    env_cfg = GuideV2ScannerTerrainEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.curriculum = None
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise ValueError("V12 collection requires generated terrain")
    generator.curriculum = False
    env_cfg.scene.terrain.max_init_terrain_level = int(generator.num_rows) - 1
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.scene.depth_camera = make_v12_depth_camera_cfg(step_dt)

    agent_cfg = copy.deepcopy(V11_TERRAIN_PPO_CFG)
    agent_cfg.device = env_cfg.sim.device
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    pin_all_terrain_rows_and_types(raw_env)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    output_dir = Path(args_cli.output_dir)
    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        load_inference_checkpoint(runner, args_cli.checkpoint)
        runner.alg.policy.reset()
        teacher_policy = runner.get_inference_policy(device=agent_cfg.device)
        frozen_actor = load_frozen_v11_actor(
            args_cli.checkpoint,
            device=agent_cfg.device,
        )
        student = None
        if args_cli.rollout_policy == "student":
            student = load_student(args_cli.student_checkpoint, agent_cfg.device)

        observations = env.get_observations()
        with torch.inference_mode():
            for _ in range(args_cli.warmup_steps):
                teacher_action = teacher_policy(observations)
                observations, _, dones, _ = env.step(teacher_action)
                runner.alg.policy.reset(dones)

            policy_observation = policy_observation_tensor(observations)
            teacher_action = teacher_policy(observations)
            manual_action = frozen_actor(policy_observation)
            equivalence_error = float((teacher_action - manual_action).abs().max())
            if equivalence_error > 1.0e-6:
                raise RuntimeError(
                    "manual frozen actor differs from RSL-RL policy: "
                    f"{equivalence_error:.9g}"
                )

        spec = camera_spec(step_dt)
        writer = DatasetShardWriter(
            output_dir,
            split=args_cli.split,
            seed=args_cli.seed,
            target_samples=args_cli.samples,
            shard_size=args_cli.shard_size,
            teacher_checkpoint_sha256=sha256_file(args_cli.checkpoint),
            camera=spec,
            metadata={
                "num_envs": args_cli.num_envs,
                "warmup_steps": args_cli.warmup_steps,
                "sample_stride": args_cli.sample_stride,
                "rollout_policy": args_cli.rollout_policy,
                "manual_actor_max_abs_error": equivalence_error,
                "manual_actor_tolerance": 1.0e-6,
                "student_checkpoint": args_cli.student_checkpoint,
            },
        )
        sensor = env.unwrapped.scene["depth_camera"]
        step_index = 0
        preview_saved = False
        with torch.inference_mode():
            while writer.sample_count < args_cli.samples:
                policy_observation = policy_observation_tensor(observations)
                teacher_action = teacher_policy(observations)
                depth = None
                if step_index % args_cli.sample_stride == 0 or student is not None:
                    depth = camera_depth_nchw(sensor)
                student_action = None
                if student is not None:
                    predicted_scan = student(depth)
                    student_observation = reconstruct_teacher_observation(
                        policy_observation[:, :48],
                        predicted_scan,
                    )
                    student_action = frozen_actor(student_observation)
                if step_index % args_cli.sample_stride == 0:
                    batch = {
                        "depth": depth,
                        "proprioception": policy_observation[:, :48],
                        "teacher_scan": policy_observation[:, 48:],
                        "teacher_action": teacher_action,
                    }
                    writer.append(batch)
                    if not preview_saved:
                        torch.save(depth[0].detach().cpu(), output_dir / "depth_preview.pt")
                        preview_saved = True
                    print(
                        f"[COLLECT] {writer.sample_count}/{args_cli.samples}",
                        flush=True,
                    )
                action = select_rollout_action(
                    args_cli.rollout_policy,
                    teacher_action,
                    student_action,
                )
                observations, _, dones, _ = env.step(action)
                runner.alg.policy.reset(dones)
                step_index += 1

        manifest = writer.finalize()
        report = {
            "status": "collection_complete",
            "output_dir": str(output_dir.resolve()),
            "sample_count": manifest["sample_count"],
            "shards": len(manifest["shards"]),
            "depth_range": manifest["statistics"]["depth"],
            "manual_actor_max_abs_error": equivalence_error,
        }
        (output_dir / "collection_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        return report
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
