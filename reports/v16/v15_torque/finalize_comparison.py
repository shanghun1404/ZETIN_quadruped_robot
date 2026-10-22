"""Build the final V15 decision from completed, recorded evaluations only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "optimization"
ORIGINAL_SHA = "e2bb747acb4737f9e73dc34446cfae692ca2564ed36560fb1f137814eea46d33"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    inputs = {
        "five_guard_filter_candidates": OUT / "final_candidate_summary.json",
        "eval542": OUT / "finetune536/evaluation/screening_summary.json",
        "eval562": OUT / "reward_finetune/evaluation/screening_summary.json",
    }
    records = {key: read(path) for key, path in inputs.items()}
    older, first, second = records.values()
    assert older["all_five_candidates_complete"]
    assert first["all_candidates_complete"] and second["all_candidates_complete"]
    assert older["control_gate"] == first["control_gate"] == second["control_gate"]
    assert older["torque_gate"] == first["torque_gate"] == second["torque_gate"]
    assert older["selected_candidate"] is None
    assert first["screening_selected_candidate"] is None
    assert second["screening_selected_candidate"] is None

    all_candidates = {**older["candidates"], **first["candidates"], **second["candidates"]}
    assert len(all_candidates) == 8
    assert all(not value["all_gates_pass"] for value in all_candidates.values())
    rows = []
    for training_job, evaluation_job, label, result in [
        (536, 542, "adapt536_g097_lpf23", first),
        (556, 562, "adapt_torque1p25_g099_lpf25", second),
    ]:
        candidate = result["candidates"][label]
        for terrain, comp in candidate["comparisons"].items():
            ratios = comp["ratios_to_baseline"]
            rows.append({
                "training_job": training_job,
                "evaluation_job": evaluation_job,
                "label": label,
                "terrain": terrain,
                "guard_fraction": candidate["config"]["guard_fraction"],
                "cutoff_hz": candidate["config"]["cutoff_hz"],
                "actor_sha256": candidate["config"]["actor_sha256"],
                "survived_envs": comp["survived_envs"],
                "corridor_kept_envs": comp["corridor_kept_envs"],
                "progress_m": comp["mean_progress_m"],
                "speed_m_s": comp["mean_forward_speed_m_s"],
                "mean_joint_rms_nm": comp["rms_mean_joint_nm"],
                "mean_abs_joint_power_w": comp["mean_abs_joint_power_w"],
                "max_estimated_limit_fraction": comp["max_estimated_limit_fraction"],
                "estimated_saturation99_fraction": comp["saturation99_fraction"],
                **{f"{key}_change_percent": 100.0 * (value - 1.0)
                   for key, value in ratios.items()},
                "worst_joint_rms_joint": comp["worst_joint_rms_ratio_joint"],
                "worst_joint_rms_change_percent": 100.0 * (comp["worst_joint_rms_ratio"] - 1.0),
                "estimated_abs_mechanical_energy_per_m_j_m": comp["estimated_abs_mechanical_energy_per_m_j_m"],
                "failed_control_gates": [key for key, passed in comp["control_gates"].items() if not passed],
                "failed_torque_gates": [key for key, passed in comp["torque_gates"].items() if not passed],
                "terrain_all_gates_pass": comp["control_pass"] and comp["torque_pass"],
                "policy_adopted": False,
            })

    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in inputs.values()}
    dump(OUT / "trained_model_comparison.json", {
        "reference": "Completed508 original339 / guard0.99 / LPF25Hz / seed42",
        "torque_segment": "Measured t>=2s, alive before first termination; implicit PD estimates",
        "control_segment": "Original full12s evaluation metrics",
        "same_unchanged_gates_verified": True,
        "source_summary_sha256": source_hashes,
        "rows": rows,
        "both_trained_policies_adopted": False,
        "holdout_submitted": False,
        "actual_solver_motor_torque_recorded": False,
        "motor_thermal_safety_proven": False,
    })
    with (OUT / "trained_model_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "|".join(value) if isinstance(value, list) else value
                             for key, value in row.items()})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    colors = {536: "#2269a9", 556: "#b84a31"}
    label_positions = {("flat", 536): (0.23, 0.23), ("flat", 556): (0.55, 0.96),
                       ("rough", 536): (0.08, 0.93), ("rough", 556): (0.54, 0.63)}
    for axis, terrain in zip(axes, ("flat", "rough")):
        for row in (item for item in rows if item["terrain"] == terrain):
            x, y = row["rms_change_percent"], row["power_change_percent"]
            axis.scatter(x, y, s=80, color=colors[row["training_job"]], zorder=3)
            annotation = (f"Train {row['training_job']} / eval {row['evaluation_job']}\n"
                          f"RMS {x:+.2f}%, |power| {y:+.2f}%\n"
                          f"80% occupancy {row['above80_change_percent']:+.2f}%\n"
                          f"{'This terrain passes' if row['terrain_all_gates_pass'] else 'This terrain fails torque gates'}")
            axis.annotate(annotation, (x, y), xytext=label_positions[(terrain, row["training_job"])],
                          textcoords="axes fraction", fontsize=8, va="top",
                          arrowprops={"arrowstyle": "-", "color": colors[row["training_job"]], "alpha": 0.45})
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlim(-12, 11)
        axis.set_ylim(-8, 21)
        axis.axhspan(-8, 0, xmin=0, xmax=12 / 23, color="#dcead8", alpha=0.7)
        axis.set_title(terrain)
        axis.set_xlabel("Mean joint RMS change vs508 (%)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean estimated |joint mechanical power| change (%)")
    fig.suptitle("V15 trained policy comparison: neither policy passes both terrains", fontsize=12)
    fig.text(0.5, 0.01, "Seed42, 32env, 12s; torque t>=2s alive. Green area only meets RMS/power gates.\n"
             "Implicit PD estimates; actual solver motor torque and thermal safety are unverified.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"trained_model_comparison.{suffix}", dpi=180)
    plt.close(fig)

    raw_files = []
    for manifest in sorted(ROOT.rglob("data_sha256.txt")):
        for line in manifest.read_text().splitlines():
            expected, source = line.split(maxsplit=1)
            copies = list(manifest.parent.rglob(Path(source).name))
            assert len(copies) == 1, (manifest, source, copies)
            actual = sha256(copies[0])
            assert actual == expected, (copies[0], expected, actual)
            raw_files.append({"hpc_source": source, "local": str(copies[0].relative_to(ROOT)), "sha256": actual})
    assert len(raw_files) == 92
    dump(OUT / "final_raw_data_verification.json", {
        "all_manifested_92_raw_SHA_match_local": True,
        "manifest_count": 7,
        "files": raw_files,
        "additional_reused508_19_hashes": "Verified onHPC in eval562 source_sha256.txt; already stored baseline data/source",
    })
    dump(OUT / "final_v15_torque_decision.json", {
        "retained_selected_policy": "Original339 model11699",
        "retained_selected_actor_sha256": ORIGINAL_SHA,
        "retained_guard_fraction": 0.99,
        "retained_cutoff_hz": 25.0,
        "original339_preserved": True,
        "all8_experimental_candidates_completed": True,
        "all8_experimental_candidates_failed_unchanged_gate": True,
        "optimization_candidate": None,
        "training536_adopted": False,
        "training556_adopted": False,
        "holdout_submitted": False,
        "further_V15_GPU_jobs_authorized": False,
        "control_gate": second["control_gate"],
        "torque_gate": second["torque_gate"],
        "source_summary_sha256": source_hashes,
        "candidate_failures": {
            label: {
                "all_gates_pass": candidate["all_gates_pass"],
                "by_terrain": {
                    terrain: {
                        "control": [key for key, passed in comp["control_gates"].items() if not passed],
                        "torque": [key for key, passed in comp["torque_gates"].items() if not passed],
                    } for terrain, comp in candidate["comparisons"].items()
                },
            } for label, candidate in all_candidates.items()
        },
        "raw_data_SHA_verified_count": len(raw_files),
        "actual_solver_motor_torque_recorded": False,
        "motor_thermal_safety_proven": False,
        "units": {"torque": "N m", "mechanical_power": "W per joint, mean abs(estimated torque * angular velocity)", "slew": "N m/s"},
        "configured_effort_limits_nm": {"HAA": 24.8, "HFE_KFE": 48.0},
        "limits_are_continuous_motor_ratings": False,
        "limitations": [
            "ImplicitActuator computed/applied torque are PD estimates before/after clipping, not measured PhysX motor torque.",
            "A12s single-seed screen does not establish continuous operation, thermal safety, or holdout generalization.",
            "Motor continuous rating, current/torque constants, resistance, cooling and gear efficiency are unavailable.",
            "Different policies/failures change the alive sample population; joint averages do not imply every joint improved.",
            "Training randomized gains240-360/6.4-9.6 differ from nominal guard300/8; evaluation actual gains300/8 match nominal.",
            "Baseline508 is reused with source/metadata/gain/delay checks; different physical GPUs do not imply bitwise identical simulation.",
        ],
    })
    print(json.dumps({"comparison_rows": len(rows), "failed_candidates": len(all_candidates),
                      "raw_data_hashes_verified": len(raw_files), "retained_policy": "Original339",
                      "selected_candidate": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
