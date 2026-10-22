"""CPU-only gate check for the predeclared sequential 20 Hz refinement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_case(path):
    report = json.loads(path.read_text())
    trace_path = path.with_name(path.stem + "_trace_audit.npz")
    with np.load(trace_path) as data:
        selected = (data["time_from_measurement_s"] >= 2 - 1e-10)[:, None] & data["alive"]
        torque, velocity = data["applied_torque_nm"][selected], data["joint_velocity_rad_s"][selected]
        limits = np.asarray([24.8 if name.endswith("HAA") else 48. for name in data["joint_names"]])
        assert torque.shape[1:] == (12,) and np.isfinite(torque).all() and np.isfinite(velocity).all()
        fractions = np.abs(torque) / limits
        stats = {"rms": float(np.mean(np.sqrt(np.mean(torque.astype(np.float64) ** 2, axis=0)))),
                 "power": float(np.mean(np.abs(torque * velocity))),
                 "above80": float(np.mean(fractions >= .8)),
                 "peak_fraction": float(fractions.max()),
                 "p99_fraction_mean": float(np.mean(np.quantile(fractions, .99, axis=0)))}
    path_rows = report["path"]["per_environment"]
    stats["corridor"] = sum(row["corridor_kept"] for row in path_rows)
    stats["heading"] = float(np.mean([row["max_abs_heading_error_deg"] for row in path_rows]))
    return report, stats, {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in (path, trace_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    config = next(item for item in manifest["candidates"] if item["label"] == args.candidate)
    control, torque_gate = manifest["screening"]["control_gate"], manifest["screening"]["torque_gate"]
    result = {"candidate": config, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "scenarios": {}}
    for scenario in ("flat", "rough"):
        base, previous, base_hash = load_case(args.baseline_dir / f"main339_{scenario}_seed42.json")
        report, now, data_hash = load_case(args.candidate_dir / f"{args.candidate}_{scenario}_seed42.json")
        audit = report["torque_audit"]
        assert audit["candidate_label"] == args.candidate
        assert audit["actual_cutoff_hz"] == audit["requested_cutoff_hz"] == config["cutoff_hz"]
        assert audit["actual_guard_fraction"] == config["guard_fraction"]
        assert report["metadata"]["actor_sha256"] == manifest["baseline"]["actor_sha256"]
        command = base["metadata"]["command_forward_m_s"]
        ratios = {key: now[key] / previous[key] for key in ("rms", "power", "above80", "peak_fraction", "p99_fraction_mean")}
        gates = {
            "survival": report["survived_envs"] >= base["survived_envs"],
            "progress": report["mean_directed_progress_m"] >= control["progress_fraction_at_least"] * base["mean_directed_progress_m"],
            "command_tracking": abs(report["mean_forward_speed_m_s"] - command) <= abs(base["mean_forward_speed_m_s"] - command) + control["absolute_command_speed_error_extra_allowance_m_s"],
            "rough_corridor": scenario != "rough" or now["corridor"] >= previous["corridor"],
            "rough_heading": scenario != "rough" or now["heading"] <= previous["heading"] + control["rough_mean_max_heading_error_extra_allowance_deg"],
            "rms_not_increased": ratios["rms"] <= torque_gate["mean_joint_rms_ratio_at_most"],
            "power_not_increased": ratios["power"] <= torque_gate["pooled_abs_power_ratio_at_most"],
            "limit_demand_improved": min(ratios["peak_fraction"], ratios["p99_fraction_mean"], ratios["above80"]) < .99,
        }
        result["scenarios"][scenario] = {"gates": gates, "ratios_to_baseline": ratios, "pass": all(gates.values()), "baseline_hashes": base_hash, "candidate_hashes": data_hash}
    result["all_gates_pass"] = all(item["pass"] for item in result["scenarios"].values())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate": args.candidate, "all_gates_pass": result["all_gates_pass"]}))
    raise SystemExit(0 if result["all_gates_pass"] else 10)


if __name__ == "__main__":
    main()
