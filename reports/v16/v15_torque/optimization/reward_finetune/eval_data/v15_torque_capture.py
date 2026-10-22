"""Read-only V15 evaluator overlay: add diagnostic traces without changing dynamics.

Run only on HPC in a Slurm GPU allocation. The original evaluator is hash checked;
it is read and executed in memory, never overwritten. Its measured JSON and trace
remain unchanged. The supplemental audit NPZ includes the flat preroll, target
guard input/output, gains actually stored in PhysX, and first-episode validity.
IsaacLab implicit applied/computed torque are estimates, not solver motor torque.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ORIGINAL = Path(os.environ.get("V15_EVALUATOR", "/home/shanghun1404/v15_large_scale/config/evaluate_v13_torque.py"))
EXPECTED_SHA256 = "3b50b31cb90b16b83f1eb2b0efa5a79ff4aa3775492f52143f079c702f005264"
source = ORIGINAL.read_text()
actual_sha256 = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
if actual_sha256 != EXPECTED_SHA256:
    raise RuntimeError(f"Unreviewed evaluator {actual_sha256}: {ORIGINAL}")


def replace_once(old: str, new: str) -> None:
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one instrumentation anchor: {old[:100]!r}")
    source = source.replace(old, new, 1)


replace_once(
    "        original_write = base_env.scene.write_data_to_sim\n",
    """        original_write = base_env.scene.write_data_to_sim
        audit_keys = ("applied_torque_nm", "computed_torque_nm", "joint_velocity_rad_s",
                      "joint_position_rad", "joint_position_target_rad", "raw_target_rad",
                      "filtered_target_rad", "alive", "first_episode_alive")
        audit = {key: [] for key in audit_keys}
        audit_alive = alive.clone()
        audit_action = base_env.action_manager.get_term("joint_pos")
        audit_action_indices = [audit_action._joint_names.index(name) for name in JOINT_NAMES]
        audit_cpu_indices = indices.detach().cpu()
        audit_kp = robot.root_physx_view.get_dof_stiffnesses()[:, audit_cpu_indices].detach().cpu().clone().numpy()
        audit_kd = robot.root_physx_view.get_dof_dampings()[:, audit_cpu_indices].detach().cpu().clone().numpy()
        audit_effort_limits = robot.root_physx_view.get_dof_max_forces()[:, audit_cpu_indices].detach().cpu().clone().numpy()
        audit_data_kp = robot.data.joint_stiffness[:, indices].detach().cpu().clone().numpy()
        audit_data_kd = robot.data.joint_damping[:, indices].detach().cpu().clone().numpy()
""",
)
replace_once(
    "            original_write()\n            if recording:\n",
    """            original_write()
            audit_values = {
                "applied_torque_nm": robot.data.applied_torque[:, indices],
                "computed_torque_nm": robot.data.computed_torque[:, indices],
                "joint_velocity_rad_s": robot.data.joint_vel[:, indices],
                "joint_position_rad": robot.data.joint_pos[:, indices],
                "joint_position_target_rad": robot.data.joint_pos_target[:, indices],
                "raw_target_rad": audit_action._raw_targets[:, audit_action_indices],
                "filtered_target_rad": audit_action._filtered_targets[:, audit_action_indices],
                "alive": alive,
                "first_episode_alive": audit_alive,
            }
            for audit_key, audit_value in audit_values.items():
                audit[audit_key].append(audit_value.detach().cpu().clone().numpy())
            if recording:
""",
)
replace_once(
    "                observations, _, dones, _ = env.step(policy(observations))\n",
    """                observations, _, dones, _ = env.step(policy(observations))
                audit_alive &= ~dones.bool()
""",
)
replace_once(
    "        output.write_text(json.dumps(report, indent=2) + \"\\n\", encoding=\"utf-8\")\n",
    """        if args_cli.trace_output:
            audit_path = Path(args_cli.trace_output).with_name(Path(args_cli.trace_output).stem + "_audit.npz")
            audit_count = (warmup_steps + measured_steps) * env_cfg.decimation
            if any(len(values) != audit_count for values in audit.values()):
                raise RuntimeError("Supplemental audit has an unexpected physics step count")
            np.savez_compressed(
                audit_path, **{key: np.stack(values) for key, values in audit.items()},
                time_from_reset_s=np.arange(audit_count, dtype=np.float64) * physics_dt,
                time_from_measurement_s=np.arange(audit_count, dtype=np.float64) * physics_dt - warmup_steps * control_dt,
                preroll=np.arange(audit_count) < warmup_steps * env_cfg.decimation,
                joint_names=np.asarray(JOINT_NAMES),
                physx_stiffness_nm_rad=audit_kp, physx_damping_nm_s_rad=audit_kd,
                physx_effort_limits_nm=audit_effort_limits,
                isaaclab_stiffness_nm_rad=audit_data_kp, isaaclab_damping_nm_s_rad=audit_data_kd,
                nominal_guard_stiffness_nm_rad=np.asarray(300.0),
                nominal_guard_damping_nm_s_rad=np.asarray(8.0),
                nominal_guard_velocity_limit_rad_s=np.asarray(50.0),
                nominal_guard_fraction=np.asarray(args_cli.effort_guard_fraction),
            )
            report["torque_audit"] = {
                "path": str(audit_path), "original_evaluator_sha256": "__ORIGINAL_HASH__",
                "capture_sha256": sha256(__file__),
                "recording_phase": "after scene.write_data_to_sim, before physics step",
                "applied_semantics": "IsaacLab ImplicitActuator estimated PD torque after clipping; not solver motor torque",
                "computed_semantics": "IsaacLab ImplicitActuator estimated PD torque before clipping, after position target guard",
                "actual_solver_motor_torque_recorded": False,
                "warmup_and_measurement_both_recorded": True,
            }
        output.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
""".replace("__ORIGINAL_HASH__", EXPECTED_SHA256),
)

exec(compile(source, str(ORIGINAL), "exec"), {"__name__": "__main__", "__file__": str(Path(__file__).resolve())})
