"""Validate the unchanged screened winner on seed43/44; retain control gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import save_fig, write_csv
from analyze_sweep import summarize_case, comparison
import matplotlib.pyplot as plt


def path_metrics(report):
    path = report["path"]["per_environment"]
    return {"corridor_kept_envs": sum(row["corridor_kept"] for row in path),
            "mean_max_heading_error_deg": float(np.mean([row["max_abs_heading_error_deg"] for row in path]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--data-dir", type=Path, default=root / "optimization/holdout_data")
    parser.add_argument("--output-dir", type=Path, default=root / "optimization")
    parser.add_argument("--selection", type=Path, default=root / "optimization/holdout_selection.json")
    parser.add_argument("--manifest", type=Path, default=root.parent.parent.parent / "config/experiments/v16_20260929/v15_torque_candidates.json")
    args = parser.parse_args()
    selection, manifest = json.loads(args.selection.read_text()), json.loads(args.manifest.read_text())
    label = selection["selected_candidate"]
    config = next(row for row in manifest["candidates"] if row["label"] == label)
    gate = manifest["screening"]["control_gate"]
    result = {"selected_candidate": label, "config": config, "seed42_screening_summary_sha256": selection["screening_summary_sha256"],
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "flat_baseline_rms_available": True, "cases": {},
        "torque_semantics": "Implicit PD estimates; solver motor output and thermal safety unavailable"}
    rows = []
    for seed in (43, 44):
        for scenario in ("flat", "rough"):
            baseline_path = args.data_dir / f"baseline_{scenario}_seed{seed}.json"
            candidate_path = args.data_dir / f"{label}_{scenario}_seed{seed}.json"
            baseline = json.loads(baseline_path.read_text())
            baseline_case = summarize_case(baseline_path)
            candidate_case = summarize_case(candidate_path)
            candidate = candidate_case["report"]
            for report, expected_actor in ((baseline, manifest["baseline"]["actor_sha256"]),
                                          (candidate, config.get("actor_sha256", manifest["baseline"]["actor_sha256"]))):
                meta = report["metadata"]
                assert meta["seed"] == seed and meta["scenario"] == scenario
                assert expected_actor and meta["actor_sha256"] == expected_actor
                assert meta["protocol"] == "fixed_v2" and meta["joint_drive"] == "physx"
                assert meta["heading_control"] == (scenario == "rough")
                assert meta["measured_s"] == 12 and meta["warmup_s"] == (2 if scenario == "flat" else 0)
            actual = candidate["torque_audit"]
            assert actual["candidate_label"] == label
            assert actual["actual_cutoff_hz"] == config["cutoff_hz"]
            assert actual["actual_guard_fraction"] == config["guard_fraction"]
            if "baseline_reuse" in manifest:
                for field, expected in baseline["metadata"].items():
                    if field not in ("actor_checkpoint", "actor_sha256"):
                        assert candidate["metadata"][field] == expected, (seed, scenario, field)
                assert actual["actual_max_delay_steps"] == baseline["torque_audit"]["actual_max_delay_steps"] == 0
            bp, cp = path_metrics(baseline), path_metrics(candidate)
            command = baseline["metadata"]["command_forward_m_s"]
            berror = abs(baseline["mean_forward_speed_m_s"] - command)
            cerror = abs(candidate["mean_forward_speed_m_s"] - command)
            control_checks = {
                "survival": candidate["survived_envs"] >= baseline["survived_envs"],
                "progress": candidate["mean_directed_progress_m"] >= gate["progress_fraction_at_least"] * baseline["mean_directed_progress_m"],
                "command_tracking": cerror <= berror + gate["absolute_command_speed_error_extra_allowance_m_s"],
                "rough_corridor": scenario == "flat" or cp["corridor_kept_envs"] >= bp["corridor_kept_envs"],
                "rough_heading": scenario == "flat" or cp["mean_max_heading_error_deg"] <= bp["mean_max_heading_error_deg"] + gate["rough_mean_max_heading_error_extra_allowance_deg"],
            }
            btorque, ctorque = baseline["actuation"], candidate["actuation"]
            names = candidate["joint_order"]
            limits = np.asarray([24.8 if name.endswith("HAA") else 48. for name in names])
            bpeak = float(np.max(np.asarray(btorque["peak_applied_nm"]) / limits))
            cpeak = float(np.max(np.asarray(ctorque["peak_applied_nm"]) / limits))
            ratios = {"progress": candidate["mean_directed_progress_m"] / baseline["mean_directed_progress_m"],
                "power": ctorque["mean_joint_power_w"] / btorque["mean_joint_power_w"],
                "p99_estimated_torque": ctorque["p99_applied_nm"] / btorque["p99_applied_nm"],
                "peak_estimated_limit_fraction": cpeak / bpeak,
                "p99_slew": ctorque["p99_applied_torque_slew_nm_s"] / btorque["p99_applied_torque_slew_nm_s"]}
            torque_checks = {"power_not_increased": ratios["power"] <= 1.0,
                "p99_or_peak_reduced_at_least1pct": min(ratios["p99_estimated_torque"], ratios["peak_estimated_limit_fraction"]) < .99,
                "p99_slew_not_increased_more_than5pct": ratios["p99_slew"] <= 1.05}
            original_compare = comparison(candidate_case, baseline_case, scenario, gate, manifest["screening"]["torque_gate"])
            assert original_compare["control_gates"] == control_checks
            torque_checks.update({"original_after2s_" + key: value for key, value in original_compare["torque_gates"].items()})
            after2s_rms_ratio = original_compare["ratios_to_baseline"]["rms"]
            passed = all(control_checks.values()) and all(torque_checks.values())
            key = f"{scenario}_seed{seed}"
            row = {"candidate": label, "scenario": scenario, "seed": seed, "passed": passed,
                "baseline_survived": baseline["survived_envs"], "candidate_survived": candidate["survived_envs"],
                "baseline_progress_m": baseline["mean_directed_progress_m"], "candidate_progress_m": candidate["mean_directed_progress_m"],
                "baseline_speed_m_s": baseline["mean_forward_speed_m_s"], "candidate_speed_m_s": candidate["mean_forward_speed_m_s"],
                "baseline_command_error_m_s": berror, "candidate_command_error_m_s": cerror,
                "baseline_abs_joint_power_w": btorque["mean_joint_power_w"], "candidate_abs_joint_power_w": ctorque["mean_joint_power_w"],
                "baseline_p99_estimated_nm": btorque["p99_applied_nm"], "candidate_p99_estimated_nm": ctorque["p99_applied_nm"],
                "baseline_peak_limit_fraction": bpeak, "candidate_peak_limit_fraction": cpeak,
                "baseline_saturation_fraction": btorque["applied_saturation_fraction"], "candidate_saturation_fraction": ctorque["applied_saturation_fraction"],
                "after2s_rms_ratio": after2s_rms_ratio,
                **{f"gate_{name}": passed_ for name, passed_ in control_checks.items()},
                **{f"ratio_{name}": value for name, value in ratios.items()}}
            rows.append(row)
            result["cases"][key] = {"metrics": row, "control_gates": control_checks, "torque_gates": torque_checks,
                "baseline_json_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(), "candidate_data_hashes": candidate_case["hashes"],
                "baseline_data_hashes": baseline_case["hashes"], "original_after2s_comparison": original_compare,
                "candidate_after2s_joint_metrics": candidate_case["phases"]["measured_after2s"]}
    result["all_holdout_gates_pass"] = all(row["passed"] for row in rows)
    result["recommended_candidate"] = label if result["all_holdout_gates_pass"] else None
    result["motor_thermal_safety_proven"] = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "holdout_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_csv(args.output_dir / "holdout_metrics.csv", rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(4)
    ticks = [f"{r['scenario']}\nseed{r['seed']}" for r in rows]
    for ax, fields, title in ((axes[0, 0], ("baseline_progress_m", "candidate_progress_m"), "Directed progress [m]"),
                              (axes[0, 1], ("baseline_command_error_m_s", "candidate_command_error_m_s"), "Absolute command speed error [m/s]"),
                              (axes[1, 0], ("baseline_abs_joint_power_w", "candidate_abs_joint_power_w"), "Mean estimated |torque x velocity| [W/joint]"),
                              (axes[1, 1], ("baseline_p99_estimated_nm", "candidate_p99_estimated_nm"), "Pooled p99 estimated torque [Nm]")):
        ax.bar(x - .17, [r[fields[0]] for r in rows], .34, label="baseline")
        ax.bar(x + .17, [r[fields[1]] for r in rows], .34, label=label)
        ax.set_title(title)
        ax.set_xticks(x, ticks)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend()
    fig.suptitle(f"V15 {label} unchanged holdout / seed43-44\n{'All acceptance gates passed' if result['all_holdout_gates_pass'] else 'Acceptance failed; candidate not adopted'} / actual motor torque and thermal safety unverified")
    fig.tight_layout(rect=(0, 0, 1, .93))
    save_fig(fig, args.output_dir / "holdout_comparison")
    print(json.dumps({"candidate": label, "all_gates_pass": result["all_holdout_gates_pass"],
        "failed": {key: [name for group in (value["control_gates"], value["torque_gates"]) for name, passed in group.items() if not passed]
                   for key, value in result["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
