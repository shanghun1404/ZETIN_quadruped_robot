"""Summarize independent V16 evaluation seeds from saved results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root):
    checkpoint = json.loads((root / "training_checkpoint.json").read_text())
    training_manifest = json.loads((root / "training_manifest.json").read_text())
    conditions = json.loads((root / "evaluation_conditions.json").read_text())
    expected_seconds = conditions["evaluation_seconds"]
    rows = []
    for path in sorted(root.glob("*/eval_seed*.json")):
        report = json.loads(path.read_text())
        if not np.isclose(report["seconds"], expected_seconds, rtol=0., atol=1e-9):
            raise ValueError(f"Evaluation duration mismatch: {path.parent.name}")
        duration = report["seconds"] - 1.
        speed = report["speed_command_m_s"]
        source_verified = None
        if path.parent.name.startswith("model_"):
            manifest = json.loads(path.with_name("manifest.json").read_text())
            source_verified = manifest["code_sha256"] == checkpoint["controller_code_sha256"]
            source_verified &= manifest["usd_sha256"] == training_manifest["usd_sha256"]
            source_verified &= report["speed_command_m_s"] == checkpoint["speed_command_m_s"]
            for key in ("force_allocator", "walk_axis_world", "gait_mode", "stance_point", "effort_fraction",
                        "contact_projection", "base_acceleration_weight", "smooth_com_reference",
                        "joint_friction", "joint_dynamic_friction", "joint_viscous_friction"):
                if key in checkpoint:
                    source_verified &= manifest[key] == checkpoint[key]
            if checkpoint["force_allocator"] == "wbc":
                source_verified &= manifest["wbc_iterations"] == checkpoint["wbc_iterations"]
                source_verified &= manifest.get("wbc_tracking_metric", "normalized") == checkpoint.get("wbc_tracking_metric", "normalized")
            if not source_verified:
                raise ValueError(f"Source/plant/evaluation condition mismatch: {path.parent.name}")
        row = {"run": path.parent.name, "seed": report["seed"],
               "controller": report["controller"], "seconds": report["seconds"], "survived": report["survived_envs"],
               "num_envs": report["num_envs"], "speed_command_m_s": speed,
               "mean_speed_m_s": float(np.mean(report["speed"])),
               "mean_progress_m": float(np.mean(report["progress_m"])),
               "mean_power_w_per_joint": float(np.mean(report["power"])),
               "mean_guard_fraction": float(np.mean(report["guard_fraction"]))}
        row["source_and_conditions_verified"] = source_verified
        row["mean_completed_steps"] = float(np.mean(report["completed_step_count"]))
        row["mean_qp_infeasible_tick_fraction"] = float(np.mean(report["qp_infeasible_tick_fraction"]))
        row["max_qp_torque_violation_nm"] = float(np.max(report["qp_max_torque_violation_nm"]))
        row["gates"] = {
            "survival_at_least_95_percent": row["survived"] / row["num_envs"] >= .95,
            "mean_speed_error_at_most_0_05_m_s": abs(row["mean_speed_m_s"] - speed) <= .05,
            "mean_progress_75_to_125_percent_of_command": .75 * speed * duration <= row["mean_progress_m"] <= 1.25 * speed * duration,
            "finite_metrics": bool(np.isfinite([row["mean_speed_m_s"], row["mean_progress_m"],
                                                row["mean_power_w_per_joint"], row["mean_guard_fraction"]]).all()),
        }
        row["gates_passed"] = all(row["gates"].values())
        rows.append(row)
    learned = [r for r in rows if r["run"].startswith("model_")]
    result = {"rows": rows, "expected_holdout_seeds": [1001, 1002, 1003],
              "evaluation_seconds": expected_seconds,
              "source_and_conditions_verified": all(r["source_and_conditions_verified"] for r in learned),
              "all_holdouts_present": len(learned) == 3 and {r["seed"] for r in learned} == {1001, 1002, 1003},
              "quality_gates_are_experiment_criteria_not_hardware_certification": True,
              "hardware_motor_continuous_rating_available": False}
    result["model_holdout_passed"] = result["all_holdouts_present"] and all(r["gates_passed"] for r in learned)
    (root / "holdout_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    table = ["# V16 independent holdout evaluation", "",
             "Execution completion and walking quality are reported separately.", "",
             "| Run | Survived | Speed (m/s) | Progress (m) | Power (W/joint) | Gates |",
             "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        table.append(f"| {r['run']} | {r['survived']}/{r['num_envs']} | {r['mean_speed_m_s']:.4f} | {r['mean_progress_m']:.4f} | {r['mean_power_w_per_joint']:.3f} | {r['gates_passed']} |")
    table += ["", f"Model holdout quality passed: **{result['model_holdout_passed']}**.",
              "", "95% survival, speed error <=0.05 m/s and progress 75–125% of commanded distance are required per seed.",
              "Torque data are explicit commands. Solver contact/friction forces and hardware thermal limits are separate."]
    (root / "README.md").write_text("\n".join(table) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
