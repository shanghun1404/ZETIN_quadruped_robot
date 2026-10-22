"""Score guard/LPF candidates only after survival, progress and tracking gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import metrics, save_fig, write_csv

SCENARIOS = ("flat", "rough")


def summarize_case(report_path: Path):
    report = json.loads(report_path.read_text())
    trace_path = report_path.with_name(report_path.stem + "_trace_audit.npz")
    with np.load(trace_path) as archive:
        trace = {key: archive[key] for key in archive.files}
    a, c, v = trace["applied_torque_nm"], trace["computed_torque_nm"], trace["joint_velocity_rad_s"]
    q, target, t = trace["joint_position_rad"], trace["joint_position_target_rad"], trace["time_from_measurement_s"]
    names = trace["joint_names"].tolist()
    limits = np.asarray([24.8 if name.endswith("HAA") else 48. for name in names])
    assert a.shape == c.shape == v.shape == (len(t), 32, 12)
    assert all(np.isfinite(x).all() for x in (a, c, v, q, target))
    assert np.all(trace["physx_stiffness_nm_rad"] == 300.)
    assert np.all(trace["physx_damping_nm_s_rad"] == 8.)
    assert np.all(trace["physx_effort_limits_nm"] == limits.astype(np.float32))
    assert trace["nominal_guard_stiffness_nm_rad"] == 300.
    assert trace["nominal_guard_damping_nm_s_rad"] == 8.
    assert trace["nominal_guard_velocity_limit_rad_s"] == 50.
    pre = 300 * (trace["filtered_target_rad"] - q) - 8 * v
    spring, damping = 300 * (target - q), -8 * v
    worst = np.maximum.reduce([np.abs(kp * spring + kd * damping) for kp in (.8, 1.2) for kd in (.8, 1.2)])
    clip = np.abs(target - trace["filtered_target_rad"]) > 1e-6
    phases = {}
    for phase, time_mask in (("measured_all", t >= -1e-10), ("measured_after2s", t >= 2 - 1e-10)):
        phases[phase] = metrics(a, c, v, time_mask[:, None] & trace["alive"], limits,
            report["metadata"]["physics_dt_s"], pre, spring + damping, worst, clip)
    original, measured = report["actuation"], phases["measured_all"]
    assert np.allclose(measured["peak_applied_nm"], original["peak_applied_nm"], atol=1e-5)
    assert np.isclose(measured["pooled_mean_abs_joint_power_w"], original["mean_joint_power_w"], rtol=2e-5)
    assert np.isclose(measured["pooled_saturation_fraction"], original["applied_saturation_fraction"], rtol=2e-5, atol=1e-6)
    path = report["path"]["per_environment"]
    total_estimated_abs_energy_j = measured["pooled_mean_abs_joint_power_w"] * 12 * report["metadata"]["physics_dt_s"] * measured["live_env_physics_samples"]
    total_progress_m = report["mean_directed_progress_m"] * report["initial_envs"]
    return {"report": report, "phases": phases, "joint_names": names,
            "torque_rms_mean_nm": float(np.mean(phases["measured_after2s"]["rms_applied_nm"])),
            "estimated_abs_mechanical_energy_per_m_j_m": total_estimated_abs_energy_j / total_progress_m if total_progress_m > 0 else None,
            "mean_max_heading_error_deg": float(np.mean([row["max_abs_heading_error_deg"] for row in path])),
            "corridor_kept_envs": sum(row["corridor_kept"] for row in path),
            "mean_max_cross_track_m": float(np.mean([row["max_abs_cross_track_m"] for row in path])),
            "hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (report_path, trace_path)}}


def comparison(candidate, baseline, scenario, control_gate, torque_gate):
    report, base = candidate["report"], baseline["report"]
    now, previous = candidate["phases"]["measured_after2s"], baseline["phases"]["measured_after2s"]
    command = base["metadata"]["command_forward_m_s"]
    error = abs(report["mean_forward_speed_m_s"] - command)
    base_error = abs(base["mean_forward_speed_m_s"] - command)
    gates = {
        "survival": report["survived_envs"] >= base["survived_envs"],
        "progress": report["mean_directed_progress_m"] >= control_gate["progress_fraction_at_least"] * base["mean_directed_progress_m"],
        "command_tracking": error <= base_error + control_gate["absolute_command_speed_error_extra_allowance_m_s"],
        "rough_corridor": scenario != "rough" or candidate["corridor_kept_envs"] >= baseline["corridor_kept_envs"],
        "rough_heading": scenario != "rough" or candidate["mean_max_heading_error_deg"] <= baseline["mean_max_heading_error_deg"] + control_gate["rough_mean_max_heading_error_extra_allowance_deg"],
    }
    ratios = {
        "rms": candidate["torque_rms_mean_nm"] / baseline["torque_rms_mean_nm"],
        "power": now["pooled_mean_abs_joint_power_w"] / previous["pooled_mean_abs_joint_power_w"],
        "above80": now["pooled_above80_fraction"] / previous["pooled_above80_fraction"],
        "slew": now["pooled_p99_slew_nm_s"] / previous["pooled_p99_slew_nm_s"],
        "peak_fraction": max(now["peak_limit_fraction"]) / max(previous["peak_limit_fraction"]),
        "p99_fraction_mean": float(np.mean(now["p99_limit_fraction"])) / float(np.mean(previous["p99_limit_fraction"])),
    }
    torque_checks = {
        "rms_not_increased": ratios["rms"] <= torque_gate["mean_joint_rms_ratio_at_most"],
        "power_not_increased": ratios["power"] <= torque_gate["pooled_abs_power_ratio_at_most"],
        "limit_demand_improved": min(ratios["peak_fraction"], ratios["p99_fraction_mean"], ratios["above80"]) < .99,
    }
    # Ranking is conditional on every control gate passing in both terrains.
    score = .4 * ratios["rms"] + .25 * ratios["power"] + .25 * ratios["above80"] + .1 * ratios["slew"]
    joint_rms_ratios = np.asarray(now["rms_applied_nm"]) / np.asarray(previous["rms_applied_nm"])
    worst_joint = int(np.argmax(joint_rms_ratios))
    return {"control_gates": gates, "torque_gates": torque_checks,
        "control_pass": all(gates.values()), "torque_pass": all(torque_checks.values()),
        "ratios_to_baseline": ratios, "ranking_score": score,
        "survived_envs": report["survived_envs"], "mean_progress_m": report["mean_directed_progress_m"],
        "mean_forward_speed_m_s": report["mean_forward_speed_m_s"], "absolute_command_error_m_s": error,
        "corridor_kept_envs": candidate["corridor_kept_envs"], "mean_max_heading_error_deg": candidate["mean_max_heading_error_deg"],
        "rms_mean_joint_nm": candidate["torque_rms_mean_nm"],
        "pooled_p99_estimated_nm": now["pooled_p99_applied_nm"], "max_estimated_limit_fraction": max(now["peak_limit_fraction"]),
        "above80_fraction": now["pooled_above80_fraction"], "saturation99_fraction": now["pooled_saturation_fraction"],
        "mean_abs_joint_power_w": now["pooled_mean_abs_joint_power_w"], "p99_torque_slew_nm_s": now["pooled_p99_slew_nm_s"],
        "worst_joint_rms_ratio": float(joint_rms_ratios[worst_joint]),
        "worst_joint_rms_ratio_joint": candidate["joint_names"][worst_joint],
        "per_joint_rms_ratios": dict(zip(candidate["joint_names"], joint_rms_ratios.tolist())),
        "baseline_estimated_abs_mechanical_energy_per_m_j_m": baseline["estimated_abs_mechanical_energy_per_m_j_m"],
        "estimated_abs_mechanical_energy_per_m_j_m": candidate["estimated_abs_mechanical_energy_per_m_j_m"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--baseline-dir", type=Path, default=root / "data")
    parser.add_argument("--sweep-dir", type=Path, default=root / "optimization" / "data")
    parser.add_argument("--manifest", type=Path, default=root.parent.parent.parent / "config/experiments/v16_20260929/v15_torque_candidates.json")
    parser.add_argument("--output-dir", type=Path, default=root / "optimization")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    baseline = {scenario: summarize_case(args.baseline_dir / f"main339_{scenario}_seed42.json") for scenario in SCENARIOS}
    result = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "control_gate": manifest["screening"]["control_gate"], "torque_gate": manifest["screening"]["torque_gate"],
        "score_weights": {"rms": .4, "abs_power": .25, "above80": .25, "p99_slew": .1},
        "torque_semantics": "Implicit PD estimates, not actual solver motor torque or thermal validation", "candidates": {}}
    rows, joint_rows, complete = [], [], []
    cases = {"baseline": baseline}
    for config in manifest["candidates"]:
        label = config["label"]
        files = [args.sweep_dir / f"{label}_{scenario}_seed42.json" for scenario in SCENARIOS]
        if not all(path.exists() for path in files):
            continue
        candidate_cases = {scenario: summarize_case(path) for scenario, path in zip(SCENARIOS, files)}
        cases[label] = candidate_cases
        comparisons = {}
        for scenario in SCENARIOS:
            item = candidate_cases[scenario]
            audit = item["report"]["torque_audit"]
            assert audit["candidate_label"] == label
            assert audit["actual_cutoff_hz"] == audit["requested_cutoff_hz"] == config["cutoff_hz"]
            assert audit["actual_guard_fraction"] == config["guard_fraction"]
            expected_actor_sha256 = config.get("actor_sha256", manifest["baseline"]["actor_sha256"])
            assert expected_actor_sha256 and item["report"]["metadata"]["actor_sha256"] == expected_actor_sha256
            assert np.isclose(audit["actual_lpf_alpha"], .005 / (.005 + 1 / (2 * np.pi * config["cutoff_hz"])))
            if "baseline_reuse" in manifest:
                for field, expected in manifest["baseline_reuse"]["metadata_by_scenario"][scenario].items():
                    assert item["report"]["metadata"][field] == expected, (label, scenario, field)
                assert audit["actual_max_delay_steps"] == manifest["baseline_reuse"]["verified_delay_steps"] == 0
            compare = comparison(item, baseline[scenario], scenario, result["control_gate"], result["torque_gate"])
            comparisons[scenario] = compare
            rows.append({"candidate": label, "guard_fraction": config["guard_fraction"], "cutoff_hz": config["cutoff_hz"],
                "scenario": scenario, **{key: value for key, value in compare.items() if not isinstance(value, dict)},
                **{f"gate_{key}": value for key, value in compare["control_gates"].items()},
                **{f"ratio_{key}": value for key, value in compare["ratios_to_baseline"].items()}})
        passed = all(item["control_pass"] and item["torque_pass"] for item in comparisons.values())
        score = float(np.mean([item["ranking_score"] for item in comparisons.values()]))
        result["candidates"][label] = {"config": config, "comparisons": comparisons, "all_gates_pass": passed,
            "actual_settings_verified": {"cutoff_hz": config["cutoff_hz"], "guard_fraction": config["guard_fraction"],
                "actor_sha256": config.get("actor_sha256", manifest["baseline"]["actor_sha256"]),
                "physx_kp_nm_rad": 300., "physx_kd_nm_s_rad": 8., "guard_velocity_limit_rad_s": 50.,
                "actual_solver_motor_torque_recorded": False},
            "mean_ranking_score": score, "data_hashes": {scenario: candidate_cases[scenario]["hashes"] for scenario in SCENARIOS}}
        complete.append(label)
    expected = [config["label"] for config in manifest["candidates"]]
    result["sequential_policy"] = manifest.get("sequential_policy")
    if result["sequential_policy"]:
        policy = result["sequential_policy"]
        first = policy["first"]
        if first in result["candidates"] and result["candidates"][first]["all_gates_pass"]:
            assert policy["stop_after_first_pass"] and complete == [first]
            expected = [first]
            result["omitted_optional_candidates"] = [policy["second_if_first_fails_any_original_gate"]]
        else:
            result["omitted_optional_candidates"] = []
    result["complete_candidates"] = complete
    result["all_candidates_complete"] = complete == expected
    passing = [label for label in complete if result["candidates"][label]["all_gates_pass"]]
    result["screening_selected_candidate"] = min(passing, key=lambda label: result["candidates"][label]["mean_ranking_score"]) if passing and result["all_candidates_complete"] else None
    result["selection_is_provisional_pending_holdout"] = result["screening_selected_candidate"] is not None
    for label, case_group in cases.items():
        for scenario, item in case_group.items():
            metrics_ = item["phases"]["measured_after2s"]
            for j, name in enumerate(item["joint_names"]):
                row = {"candidate": label, "scenario": scenario, "joint": name}
                row.update({key: value[j] for key, value in metrics_.items() if isinstance(value, list)})
                joint_rows.append(row)
    (out / "screening_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_csv(out / "screening_metrics.csv", rows)
    write_csv(out / "screening_joint_metrics.csv", joint_rows)
    if complete:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        labels = list(cases)
        names = baseline["flat"]["joint_names"]
        x = np.arange(12)
        for i, scenario in enumerate(SCENARIOS):
            for j, (field, title) in enumerate((("rms_limit_fraction", "RMS / configured limit"),
                                               ("above_80pct_fraction", "Time fraction >= 80% limit"),
                                               ("mean_abs_mechanical_power_w", "Mean |torque x velocity| [W]"))):
                ax = axes[i, j]
                width = .8 / len(labels)
                for k, label in enumerate(labels):
                    values = cases[label][scenario]["phases"]["measured_after2s"][field]
                    ax.bar(x + (k - (len(labels) - 1) / 2) * width, values, width=width, label=label)
                ax.set_title(scenario + " / " + title)
                ax.set_xticks(x, names, rotation=65, fontsize=7)
                ax.grid(axis="y", alpha=.2)
        axes[0, 0].legend(fontsize=8)
        experiment_title = "V15 torque L2 reward adaptation / guard0.99 LPF25" if "reward_training_inputs" in manifest else "V15 conservative guard/LPF sweep"
        fig.suptitle(experiment_title + " / alive t>=2s / seed42\nCandidate acceptance requires survival, progress, speed tracking and rough direction gates")
        fig.tight_layout(rect=(0, 0, 1, .94))
        save_fig(fig, out / "screening_torque_comparison")
    print(json.dumps({"complete": complete, "provisional_selected": result["screening_selected_candidate"],
        "candidates": {label: {"pass": value["all_gates_pass"], "score": value["mean_ranking_score"],
            "control_failures": {scenario: [key for key, passed in compare["control_gates"].items() if not passed]
                                 for scenario, compare in value["comparisons"].items()}}
            for label, value in result["candidates"].items()}}, indent=2))


if __name__ == "__main__":
    main()
