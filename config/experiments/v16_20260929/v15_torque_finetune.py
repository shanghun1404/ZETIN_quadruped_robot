"""Prepared, unsubmitted V15 motor-adaptation pilot; originals stay hash checked."""
from __future__ import annotations

import argparse
import ast
import hashlib
import math
import os
from pathlib import Path
import sys

FIXED_LR = 3.375e-5
TRAINER_SHA256 = "3da8cdcefa600f851795d3d48cdd30a6172b55070556ff8d888971057a1391d2"
ACTOR = "/home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt"
INPUTS = {
    "actor_checkpoint": (ACTOR, "e2bb747acb4737f9e73dc34446cfae692ca2564ed36560fb1f137814eea46d33"),
    "student_checkpoint": ("/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt", "74c0c15644aec802c2637e5054286ea261e0fd6060637506eee61776af3b6f05"),
    "usd_path": ("/home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda", "864f4b11fc24af1099acf292ce5358192294426928387c6d903cba41d45050ad"),
}
SOURCE_HASHES = {
    "v15_env_cfg": "9d7630498116cff12bde9f05f38a7957ea8dd176a2a42d00aaaa73bb1d1e8672",
    "v14_leg_env_cfg": "2cc074a3419189b1b95378d025106c0a189a68d4a58710967071f8520d60a3c4",
    "v13_torque_actions": "909f240bd141f5d9e0e676f75227a89aa17b98ac8722d2949ae0c3b767173066",
    "v13_torque_mdp": "b156a82fa97f5d5bc2590dd0ca63ed5256605b1548f4d4d1caf20c79f0f98297",
    "v15_math": "36d64aca24c0eb339d0dcfcef74670e2e71fcc737129ccda6f848e183471e2d3",
}


def synchronize_fixed_lr(algorithm):
    """Keep restored optimizer moments; explicitly reset both LR owners."""
    groups = algorithm.optimizer.param_groups
    if not groups or any(not math.isfinite(float(group["lr"])) or float(group["lr"]) <= 0 for group in groups):
        raise ValueError("Finite positive LR and at least one restored optimizer group required")
    before = {"algorithm_learning_rate_before": float(algorithm.learning_rate),
              "schedule_before": algorithm.schedule,
              "optimizer_group_lr_before": [float(group["lr"]) for group in groups]}
    algorithm.learning_rate = FIXED_LR
    algorithm.schedule = "fixed"
    for group in groups:
        group["lr"] = FIXED_LR
    before.update({"algorithm_learning_rate_after": float(algorithm.learning_rate),
                   "schedule_after": algorithm.schedule,
                   "optimizer_group_lr_after": [float(group["lr"]) for group in groups],
                   "algorithm_optimizer_lr_match": all(group["lr"] == algorithm.learning_rate == FIXED_LR for group in groups),
                   "restored_optimizer_state_entries": len(algorithm.optimizer.state)})
    assert before["algorithm_optimizer_lr_match"]
    return before


def build_source(original: Path):
    if hashlib.sha256(original.read_bytes()).hexdigest() != TRAINER_SHA256:
        raise RuntimeError("Unreviewed original V15 trainer")
    source = original.read_text()

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"Expected one trainer anchor: {old!r}")
        source = source.replace(old, new, 1)

    replace('parser.add_argument("--num_envs", type=int, default=512)', 'parser.add_argument("--num_envs", type=int, default=1024)')
    replace('parser.add_argument("--max_iterations", type=int, default=6000)', 'parser.add_argument("--max_iterations", type=int, default=250)')
    replace('    env_cfg = GuideV15LocomotionEnvCfg()\n', '''    assert args_cli.resume, "Motor adaptation restores339 weights and optimizer"
    loaded_source_hashes = {}
    for module_name, expected_hash in FINETUNE_SOURCE_HASHES.items():
        module_path = sys.modules[module_name].__file__
        loaded_source_hashes[module_name] = file_sha256(module_path)
        if loaded_source_hashes[module_name] != expected_hash:
            raise RuntimeError(f"Unreviewed source {module_name}")
    env_cfg = GuideV15LocomotionEnvCfg()
''')
    replace('    env_cfg.actions.joint_pos.max_delay_steps = args_cli.max_delay_steps\n', '''    env_cfg.actions.joint_pos.max_delay_steps = args_cli.max_delay_steps
    env_cfg.actions.joint_pos.cutoff_hz = 23.0
    env_cfg.actions.joint_pos.effort_guard_fraction = 0.97
    assert env_cfg.sim.dt == 0.005 and env_cfg.decimation == 4
    assert env_cfg.episode_length_s == 20.0 and args_cli.max_delay_steps == 3
    assert env_cfg.terminations.physics_diverged is not None
    assert env_cfg.rewards.torque_pressure is None and env_cfg.rewards.joint_power is None
''')
    replace('    agent_cfg.experiment_name = "guide_v15_physx_pd"', '    agent_cfg.experiment_name = "guide_v15_torque_finetune"')
    replace('    agent_cfg.run_name = "v14_m5700_physx_pd_lane_stairs"', '    agent_cfg.run_name = "339_g097_lpf23_lr3p375e-5"')
    replace('    agent_cfg.save_interval = 100\n', '''    agent_cfg.save_interval = 100
    agent_cfg.algorithm.learning_rate = FINETUNE_FIXED_LR
    agent_cfg.algorithm.schedule = "fixed"
    assert agent_cfg.num_steps_per_env == 24
''')
    replace('        "action_lpf_hz": 25.0,', '        "action_lpf_hz": float(env_cfg.actions.joint_pos.cutoff_hz),')
    replace('        "schema": 1,\n', '''        "schema": 1,
        "prepared_motor_adaptation": True,
        "original_trainer_sha256": FINETUNE_TRAINER_SHA256,
        "overlay_sha256": file_sha256(__file__),
        "loaded_source_hashes": loaded_source_hashes,
        "reward_changes": False,
        "actual_solver_motor_torque_recorded": False,
        "motor_thermal_safety_proven": False,
''')
    replace('        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)\n', '''        lr_contract = finetune_sync_lr(runner.alg)
        action = raw_env.unwrapped.action_manager.get_term("joint_pos")
        assert action.cfg.cutoff_hz == 23.0 and action.cfg.effort_guard_fraction == 0.97
        assert action.cfg.max_delay_steps == 3
        contract = {
            **lr_contract, "resume_optimizer": True,
            "starting_learning_iteration": runner.current_learning_iteration,
            "num_learning_iterations": agent_cfg.max_iterations,
            "num_envs": args_cli.num_envs, "seed": args_cli.seed,
            "actual_cutoff_hz": float(action.cfg.cutoff_hz),
            "actual_lpf_alpha": float(action._alpha),
            "actual_guard_fraction": float(action.cfg.effort_guard_fraction),
            "max_delay_steps": action.cfg.max_delay_steps,
            "physics_dt_s": float(env_cfg.sim.dt), "control_dt_s": env_cfg.sim.dt * env_cfg.decimation,
            "rollout_steps_per_env": agent_cfg.num_steps_per_env,
            "episode_length_s": float(env_cfg.episode_length_s),
            "gain_scale_range": list(V15_GAIN_SCALE_RANGE),
            "actual_stiffness_min_max": [float(stiffness.min()), float(stiffness.max())],
            "actual_damping_min_max": [float(raw_env.unwrapped.scene["robot"].data.joint_damping.min()), float(raw_env.unwrapped.scene["robot"].data.joint_damping.max())],
            "finite_wrapper": type(env).__name__, "physics_diverged_termination": True,
            "torque_pressure_reward_enabled": False, "joint_power_reward_enabled": False,
            "loaded_source_hashes": loaded_source_hashes,
        }
        (log_dir / "finetune_actual_contract.json").write_text(json.dumps(contract, indent=2) + "\\n")
        print("[INFO] V15 fixed LR contract " + json.dumps(contract), flush=True)
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
''')
    replace('    finally:\n        env.close()\n', '''    finally:
        # Synchronous checkpoint writers finish before the optional Kit teardown.
        writer = getattr(locals().get("runner"), "writer", None)
        if writer is not None:
            writer.flush()
            writer.close()
    torch.cuda.synchronize()
    checkpoint = log_dir / f"model_{runner.current_learning_iteration}.pt"
    if not checkpoint.is_file():
        raise RuntimeError("Final checkpoint missing")
    if not all(bool(torch.isfinite(value).all()) for value in runner.alg.policy.state_dict().values()):
        raise RuntimeError("Nonfinite final policy")
    policy_std = runner.alg.policy.std
    if not bool((policy_std > 0).all()):
        raise RuntimeError("Nonpositive final action standard deviation")
    completed = {"training_completed": True, "checkpoint": str(checkpoint),
                 "checkpoint_sha256": file_sha256(checkpoint),
                 "last_learning_iteration": runner.current_learning_iteration,
                 "policy_std_min_max": [float(policy_std.min()), float(policy_std.max())],
                 "nonfinite_steps": int(getattr(env, "nonfinite_steps", 0)),
                 "actual_contract_sha256": file_sha256(str(log_dir / "finetune_actual_contract.json")),
                 "behavior_and_torque_gates_evaluated": False, "motor_thermal_safety_proven": False}
    (log_dir / "run_completed.json").write_text(json.dumps(completed, indent=2) + "\\n")
    print("[INFO] V15 pilot saved " + json.dumps(completed), flush=True)
''')
    replace('    simulation_app.close()\n', '    sys.stdout.flush()\n    sys.stderr.flush()\n    os._exit(0)\n')
    return source


def main():
    original = Path(os.environ.get("V15_TRAINER", "/home/shanghun1404/v15_large_scale/config/train_v15_locomotion.py"))
    source = build_source(original)
    ast.parse(source)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training must run in the HPC Slurm allocation")
    preflight = argparse.ArgumentParser(add_help=False)
    for name in INPUTS:
        preflight.add_argument("--" + name.replace("_", "-"), required=True)
    preflight.add_argument("--resume", action="store_true")
    preflight.add_argument("--num_envs", type=int, default=1024)
    preflight.add_argument("--max_iterations", type=int, default=250)
    preflight.add_argument("--max-delay-steps", type=int, default=3)
    args, _ = preflight.parse_known_args()
    if not args.resume or not 1 <= args.num_envs <= 1024 or not 1 <= args.max_iterations <= 500 or args.max_delay_steps != 3:
        raise ValueError("Prepared pilot requires resume, <=1024 envs, <=500 iterations and existing delay0-3")
    for name, (expected_path, expected_hash) in INPUTS.items():
        path = Path(getattr(args, name))
        if path.resolve() != Path(expected_path).resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Unreviewed frozen input {name}")
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
                 "FINETUNE_FIXED_LR": FIXED_LR, "FINETUNE_TRAINER_SHA256": TRAINER_SHA256,
                 "FINETUNE_SOURCE_HASHES": SOURCE_HASHES, "finetune_sync_lr": synchronize_fixed_lr}
    exec(compile(source, str(original), "exec"), namespace)


if __name__ == "__main__":
    main()
