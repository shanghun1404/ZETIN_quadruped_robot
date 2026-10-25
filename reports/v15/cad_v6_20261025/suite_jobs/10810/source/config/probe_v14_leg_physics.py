"""Measure neutral standing and forced cross-leg clearance on one robot USD."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--self-collisions", action="store_true")
parser.add_argument("--mode", choices=("stand", "cross"), required=True)
parser.add_argument("--haa-amplitude", type=float, default=0.8)
parser.add_argument("--capsule-radius-m", type=float, default=0.02)
parser.add_argument("--filter-leg-terrain", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from guide_robot_v2_cfg import GUIDE_ROBOT_V2_CFG
from v14_clearance import interleg_clearances_m
from v14_terrain_filter import author_v14_ground_filter


PHYSICS_DT_S = 0.005
SETTLE_STEPS = 400
RAMP_STEPS = 200
HOLD_STEPS = 200
LEGS = ("FL", "FR", "RL", "RR")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if args.haa_amplitude <= 0 or not np.isfinite(args.haa_amplitude):
        raise ValueError("haa amplitude must be finite and positive")
    if not 0.01 <= args.capsule_radius_m <= 0.025:
        raise ValueError("capsule radius must be in [0.01, 0.025] m")
    if args.filter_leg_terrain and not args.self_collisions:
        raise ValueError("V14 terrain filtering requires self collisions")
    if not Path(args.usd_path).is_file():
        raise FileNotFoundError(args.usd_path)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    patterns = list(itertools.product((-1.0, 1.0), repeat=4)) if args.mode == "cross" else [(0.0,) * 4]
    num_envs = len(patterns)
    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT_S, device=args.device))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    robot_cfg = copy.deepcopy(GUIDE_ROBOT_V2_CFG)
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.usd_path = args.usd_path
    robot_cfg.spawn.articulation_props.enabled_self_collisions = args.self_collisions
    robot_cfg.init_state.pos = (0.0, 0.0, 0.552)
    @configclass
    class ProbeSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = robot_cfg

    scene = InteractiveScene(
        ProbeSceneCfg(
            num_envs=num_envs,
            env_spacing=2.0,
            replicate_physics=not args.filter_leg_terrain,
        )
    )
    if args.filter_leg_terrain:
        author_v14_ground_filter(scene.stage, scene.env_prim_paths)
    robot = scene["robot"]
    sim.reset()
    scene.reset()

    joint_lookup = {name: index for index, name in enumerate(robot.joint_names)}
    haa_indices = [joint_lookup[f"{leg}_HAA"] for leg in LEGS]
    signs = torch.tensor(patterns, dtype=torch.float32, device=robot.device)
    default_targets = robot.data.default_joint_pos.clone()
    if default_targets.shape[0] == 1 and num_envs > 1:
        default_targets = default_targets.expand(num_envs, -1).clone()
    if default_targets.shape[0] != num_envs:
        raise ValueError("Articulation environment count differs from probe patterns")
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=robot.device))
    root_z_initial = robot.data.root_pos_w[:, 2].detach().cpu().tolist()
    if any(abs(value - robot_cfg.init_state.pos[2]) > 1e-4 for value in root_z_initial):
        raise RuntimeError(f"probe root was not reset to configured height: {root_z_initial}")

    min_clearance = np.full(num_envs, np.inf)
    negative_samples = np.zeros(num_envs, dtype=np.int64)
    measured_samples = 0
    root_z_min = np.full(num_envs, np.inf)
    root_z_max = np.full(num_envs, -np.inf)
    final_joint_haa = None
    total_steps = SETTLE_STEPS + (RAMP_STEPS + HOLD_STEPS if args.mode == "cross" else 0)

    for step in range(total_steps):
        target = default_targets.clone()
        if step >= SETTLE_STEPS and args.mode == "cross":
            fraction = min((step - SETTLE_STEPS + 1) / RAMP_STEPS, 1.0)
            target[:, haa_indices] += fraction * args.haa_amplitude * signs
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(PHYSICS_DT_S)

        body_pos = robot.data.body_pos_w.detach().cpu().numpy()
        body_quat = robot.data.body_quat_w.detach().cpu().numpy()
        clearance = interleg_clearances_m(
            body_pos, body_quat, robot.body_names, radius_m=args.capsule_radius_m
        )
        if not np.isfinite(clearance).all():
            raise ValueError("non-finite V14 leg clearance")
        sample_min = clearance.min(axis=1)
        min_clearance = np.minimum(min_clearance, sample_min)
        negative_samples += (sample_min < 0).astype(np.int64)
        measured_samples += 1

        root_z = robot.data.root_pos_w[:, 2].detach().cpu().numpy()
        if not np.isfinite(root_z).all():
            raise ValueError("non-finite V14 root height")
        root_z_min = np.minimum(root_z_min, root_z)
        root_z_max = np.maximum(root_z_max, root_z)
        final_joint_haa = robot.data.joint_pos[:, haa_indices].detach().cpu().numpy()

    report = {
        "mode": args.mode,
        "usd_path": args.usd_path,
        "usd_sha256": sha256(args.usd_path),
        "self_collisions_enabled": args.self_collisions,
        "leg_terrain_filtered": args.filter_leg_terrain,
        "physics_dt_s": PHYSICS_DT_S,
        "num_envs": num_envs,
        "measured_steps": measured_samples,
        "haa_amplitude_rad": args.haa_amplitude if args.mode == "cross" else 0.0,
        "capsule_radius_m": args.capsule_radius_m,
        "haa_sign_patterns": patterns,
        "minimum_interleg_capsule_gap_m": float(min_clearance.min()),
        "minimum_gap_by_env_m": min_clearance.tolist(),
        "negative_clearance_sample_fraction": (negative_samples / measured_samples).tolist(),
        "root_z_initial_m": root_z_initial,
        "root_z_min_m": root_z_min.tolist(),
        "root_z_max_m": root_z_max.tolist(),
        "root_z_final_m": robot.data.root_pos_w[:, 2].detach().cpu().tolist(),
        "haa_final_rad": final_joint_haa.tolist(),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        import omni.kit.app

        omni.kit.app.get_app().post_uncancellable_quit(1)
        simulation_app.close()
        raise SystemExit(1)
