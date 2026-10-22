"""Evaluate a V15-R recovery policy: success from falls, or staying up from a standing start.

Success = upright (<20 deg tilt) at standing height (0.35-0.52 m root) held for the
final ``--hold-seconds`` of the rollout. Observation noise is off; PD-gain and
target-delay randomization stay on, as in training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--start", choices=("fallen", "standing"), default="fallen")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--hold-seconds", type=float, default=1.0)
parser.add_argument("--video-dir", help="Record --video-env-ids into this new directory (needs --enable_cameras)")
parser.add_argument("--video-env-ids", type=int, nargs="+", default=[])
parser.add_argument("--video-view", choices=("side", "rear_quarter"), default="side")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from v15_math import is_recovered, tilt_rad  # noqa: E402
from v15_recovery_env_cfg import GuideV15RecoveryEnvCfg, root_height  # noqa: E402
from v15_recovery_ppo_cfg import V15_RECOVERY_PPO_CFG  # noqa: E402

TASK_ID = "Isaac-Recovery-Guide-RobotV2-v15-eval"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not 0.0 < args_cli.hold_seconds < args_cli.seconds:
        raise ValueError("hold-seconds must be shorter than the rollout")
    if bool(args_cli.video_dir) != bool(args_cli.video_env_ids):
        raise ValueError("video-dir and video-env-ids must be supplied together")
    env_cfg = GuideV15RecoveryEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd_path
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.seconds + 5.0  # no timeout reset inside the rollout
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.reset_robot.params["fallen_fraction"] = 1.0 if args_cli.start == "fallen" else 0.0
    if args_cli.video_dir:
        # The training scene has no light; add one for the observer camera only.
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg

        env_cfg.scene.video_light = AssetBaseCfg(
            prim_path="/World/video_light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)))
    gym.register(id=TASK_ID, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)
    env = RslRlVecEnvWrapper(gym.make(TASK_ID, cfg=env_cfg))
    video = None
    try:
        agent_cfg = copy.deepcopy(V15_RECOVERY_PPO_CFG)
        agent_cfg.device = env_cfg.sim.device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint, load_optimizer=False, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        base = env.unwrapped
        robot = base.scene["robot"]
        steps = round(args_cli.seconds / base.step_dt)
        hold_steps = round(args_cli.hold_seconds / base.step_dt)
        first_recovered = torch.full((base.num_envs,), -1.0, device=base.device)
        held = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
        final_hold = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
        diverged = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        initial_tilt = tilt_rad(robot.data.projected_gravity_b).clone()
        if args_cli.video_dir:
            from v14_video import RolloutVideos

            video = RolloutVideos(base.sim, robot, args_cli.video_dir, args_cli.video_env_ids,
                                  view=args_cli.video_view, fps=1.0 / (2 * base.step_dt))
        observations = env.get_observations()
        with torch.inference_mode():
            for step in range(steps):
                if video is not None and step % 2 == 0:
                    video.capture(~diverged)
                observations, _, dones, _ = env.step(policy(observations))
                # Only the physics_diverged termination can fire before the 11 s timeout.
                diverged |= dones.bool()
                up = is_recovered(robot.data.projected_gravity_b, root_height(base))
                held = torch.where(up, held + 1, torch.zeros_like(held))
                newly = (first_recovered < 0) & (held >= round(0.5 / base.step_dt))
                first_recovered[newly] = (step + 1) * base.step_dt
                if step >= steps - hold_steps:
                    final_hold &= up
        final_hold &= ~diverged
        first_recovered[diverged] = -1.0
        times = first_recovered[first_recovered >= 0]
        report = {
            "start": args_cli.start,
            "success_count": int(final_hold.sum()),
            "num_envs": base.num_envs,
            "success_rate": float(final_hold.float().mean()),
            "median_time_to_recover_s": float(times.median()) if len(times) else None,
            "recovered_at_least_once": int((first_recovered >= 0).sum()),
            "physics_diverged_resets": int(diverged.sum()),
            "per_environment": [
                {"env_id": env_id, "initial_tilt_deg": float(initial_tilt[env_id].rad2deg()),
                 "recovered_at_end": bool(final_hold[env_id]), "diverged": bool(diverged[env_id]),
                 "first_recovery_s": float(first_recovered[env_id]) if first_recovered[env_id] >= 0 else None}
                for env_id in range(base.num_envs)
            ],
            "mean_initial_tilt_deg": float(initial_tilt.rad2deg().mean()),
            "metadata": {
                "checkpoint": args_cli.checkpoint, "checkpoint_sha256": sha256(args_cli.checkpoint),
                "usd_path": args_cli.usd_path, "usd_sha256": sha256(args_cli.usd_path),
                "seed": args_cli.seed, "seconds": steps * base.step_dt, "hold_seconds": hold_steps * base.step_dt,
                "criterion": "tilt < 20 deg and 0.35 m < root height < 0.52 m, held; first recovery = 0.5 s held",
            },
        }
        if video is not None:
            for stream in video.streams.values():
                stream["writer"].close()  # finish the mp4 files before the report records them
            report["videos"] = video.metadata()
            report["video_env_outcomes"] = {
                env_id: {"recovered_at_end": bool(final_hold[env_id]),
                         "first_recovery_s": float(first_recovered[env_id]) if first_recovered[env_id] >= 0 else None,
                         "initial_tilt_deg": float(initial_tilt[env_id].rad2deg())}
                for env_id in args_cli.video_env_ids
            }
        output = Path(args_cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if video is not None:
            try:
                video.close()
            except Exception as error:  # replicator detach failed after the files were complete (job 489)
                print(f"[WARN] observer teardown failed: {error}", flush=True)
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
