"""Reproduce the V15 torque audit from HPC traces. No physics is run locally."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = ("flat", "stairs_up", "rough")
PHASES = ("preroll", "measured_all", "measured_first2s", "measured_after2s")
SEMANTICS = "IsaacLab ImplicitActuator estimated PD after clipping; actual solver motor torque is unavailable"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(applied, computed, velocity, selected, limits, dt, nominal_pre, nominal_post, worst_gain, clipped):
    a, c, v = applied[selected], computed[selected], velocity[selected]
    if not len(a):
        return None
    fractions = np.abs(a) / limits
    saturation = np.abs(a) >= np.float32(.99) * limits.astype(np.float32)
    consecutive = selected[1:] & selected[:-1]
    slew = np.abs(np.diff(applied, axis=0))[consecutive] / dt
    power = a * v
    output = dict(
        live_env_physics_samples=int(selected.sum()),
        peak_applied_nm=np.max(np.abs(a), axis=0).tolist(),
        rms_applied_nm=np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=0)).tolist(),
        p99_applied_nm=np.quantile(np.abs(a), .99, axis=0).tolist(),
        peak_computed_nm=np.max(np.abs(c), axis=0).tolist(),
        rms_computed_nm=np.sqrt(np.mean(c.astype(np.float64) ** 2, axis=0)).tolist(),
        p99_computed_nm=np.quantile(np.abs(c), .99, axis=0).tolist(),
        peak_limit_fraction=np.max(fractions, axis=0).tolist(),
        rms_limit_fraction=np.sqrt(np.mean(fractions.astype(np.float64) ** 2, axis=0)).tolist(),
        p99_limit_fraction=np.quantile(fractions, .99, axis=0).tolist(),
        above_80pct_fraction=np.mean(fractions >= .8, axis=0).tolist(),
        above_90pct_fraction=np.mean(fractions >= .9, axis=0).tolist(),
        saturation_99pct_fraction=np.mean(saturation, axis=0).tolist(),
        over_limit_computed_fraction=np.mean(np.abs(c) > limits, axis=0).tolist(),
        actuator_clip_fraction=np.mean(np.abs(c - a) > 1e-5, axis=0).tolist(),
        guard_target_clip_fraction=np.mean(clipped[selected], axis=0).tolist(),
        peak_nominal_preguard_nm=np.max(np.abs(nominal_pre[selected]), axis=0).tolist(),
        rms_nominal_preguard_nm=np.sqrt(np.mean(nominal_pre[selected].astype(np.float64) ** 2, axis=0)).tolist(),
        p99_nominal_preguard_nm=np.quantile(np.abs(nominal_pre[selected]), .99, axis=0).tolist(),
        preguard_over_limit_fraction=np.mean(np.abs(nominal_pre[selected]) > limits, axis=0).tolist(),
        nominal_postguard_peak_nm=np.max(np.abs(nominal_post[selected]), axis=0).tolist(),
        gain_corner_counterfactual_over_limit_fraction=np.mean(worst_gain[selected] > limits, axis=0).tolist(),
        gain_corner_counterfactual_peak_ratio=np.max(worst_gain[selected] / limits, axis=0).tolist(),
        mean_abs_mechanical_power_w=np.mean(np.abs(power), axis=0).tolist(),
        mean_positive_mechanical_power_w=np.mean(np.maximum(power, 0), axis=0).tolist(),
        mean_negative_mechanical_power_w=np.mean(np.minimum(power, 0), axis=0).tolist(),
        p99_abs_mechanical_power_w=np.quantile(np.abs(power), .99, axis=0).tolist(),
        p99_torque_slew_nm_s=np.quantile(slew, .99, axis=0).tolist() if len(slew) else [0.] * applied.shape[-1],
        peak_torque_slew_nm_s=np.max(slew, axis=0).tolist() if len(slew) else [0.] * applied.shape[-1],
        pooled_p99_applied_nm=float(np.quantile(np.abs(a), .99)),
        pooled_saturation_fraction=float(np.mean(saturation)),
        pooled_above80_fraction=float(np.mean(fractions >= .8)),
        pooled_over_limit_computed_fraction=float(np.mean(np.abs(c) > limits)),
        pooled_guard_clip_fraction=float(np.mean(clipped[selected])),
        pooled_preguard_over_limit_fraction=float(np.mean(np.abs(nominal_pre[selected]) > limits)),
        pooled_gain_corner_counterfactual_over_limit_fraction=float(np.mean(worst_gain[selected] > limits)),
        pooled_mean_abs_joint_power_w=float(np.mean(np.abs(power))),
        pooled_p99_slew_nm_s=float(np.quantile(slew, .99)) if len(slew) else 0.,
        per_robot_mean_sum_abs_mechanical_power_w=float(np.mean(np.sum(np.abs(power), axis=-1))),
    )
    return output


def save_fig(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    output = args.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result = {"semantics": SEMANTICS, "actual_solver_motor_torque_recorded": False, "scenarios": {}}
    per_joint_rows, per_env_rows, series_rows, environment_rows = [], [], [], []
    files = []
    for scenario in SCENARIOS:
        stem = f"main339_{scenario}_seed42"
        report_path, trace_path = args.data_dir / f"{stem}.json", args.data_dir / f"{stem}_trace_audit.npz"
        report = json.loads(report_path.read_text())
        with np.load(trace_path) as archive:
            trace = {key: archive[key] for key in archive.files}
        for path in (report_path, trace_path, args.data_dir / f"{stem}_trace.npz", args.data_dir / f"{stem}_trajectory.npz"):
            files.append({"file": path.name, "sha256": digest(path), "bytes": path.stat().st_size})
        applied, computed = trace["applied_torque_nm"], trace["computed_torque_nm"]
        velocity, q, target = trace["joint_velocity_rad_s"], trace["joint_position_rad"], trace["joint_position_target_rad"]
        alive = trace["alive"].astype(bool)
        t, preroll = trace["time_from_measurement_s"], trace["preroll"].astype(bool)
        names, dt = trace["joint_names"].tolist(), report["metadata"]["physics_dt_s"]
        limits = np.asarray([24.8 if name.endswith("HAA") else 48. for name in names])
        assert applied.shape == computed.shape == velocity.shape == (len(t), 32, 12)
        assert alive.shape == (len(t), 32)
        assert all(np.isfinite(x).all() for x in (applied, computed, velocity, q, target))
        assert np.allclose(np.diff(t), dt)
        assert np.array_equal(trace["physx_effort_limits_nm"], np.broadcast_to(limits, (32, 12)).astype(np.float32))
        nominal_pre = 300. * (trace["filtered_target_rad"] - q) - 8. * velocity
        spring, damper = 300. * (target - q), -8. * velocity
        nominal_post = spring + damper
        # Algebraic sensitivity only: observed q/v/targets fixed, no trajectory prediction.
        worst_gain = np.maximum.reduce([np.abs(kp * spring + kd * damper) for kp in (.8, 1.2) for kd in (.8, 1.2)])
        clipped = np.abs(target - trace["filtered_target_rad"]) > 1e-6
        phase_masks = {
            "preroll": preroll[:, None] & trace["first_episode_alive"],
            "measured_all": (~preroll)[:, None] & alive,
            "measured_first2s": ((t >= -1e-10) & (t < 2. - 1e-10))[:, None] & alive,
            "measured_after2s": (t >= 2. - 1e-10)[:, None] & alive,
        }
        phases = {}
        for phase, selected in phase_masks.items():
            values = metrics(applied, computed, velocity, selected, limits, dt, nominal_pre, nominal_post, worst_gain, clipped)
            phases[phase] = values
            if values is None:
                continue
            for j, name in enumerate(names):
                row = {"scenario": scenario, "phase": phase, "joint": name, "limit_nm": limits[j], "live_env_physics_samples": values["live_env_physics_samples"]}
                row.update({key: value[j] for key, value in values.items() if isinstance(value, list)})
                per_joint_rows.append(row)
            for env in range(32):
                mask = selected[:, env]
                if not mask.any():
                    continue
                for j, name in enumerate(names):
                    x = applied[mask, env, j]
                    per_env_rows.append({"scenario": scenario, "phase": phase, "env_id": env, "joint": name,
                                         "valid_seconds": float(mask.sum() * dt), "peak_nm": float(np.abs(x).max()),
                                         "rms_nm": float(np.sqrt(np.mean(x.astype(float) ** 2))),
                                         "p99_nm": float(np.quantile(np.abs(x), .99)),
                                         "fraction_ge80_limit": float(np.mean(np.abs(x) >= .8 * limits[j])),
                                         "fraction_ge99_limit": float(np.mean(np.abs(x) >= np.float32(.99) * np.float32(limits[j])))})
        # Verify independent NumPy metrics reproduce the existing evaluator summary.
        measured = phases["measured_all"]
        original = report["actuation"]
        assert np.allclose(measured["peak_applied_nm"], original["peak_applied_nm"], atol=1e-5)
        for audit_key, original_key in (("pooled_p99_applied_nm", "p99_applied_nm"),
                                        ("pooled_mean_abs_joint_power_w", "mean_joint_power_w"),
                                        ("pooled_saturation_fraction", "applied_saturation_fraction"),
                                        ("pooled_p99_slew_nm_s", "p99_applied_torque_slew_nm_s")):
            assert np.isclose(measured[audit_key], original[original_key], rtol=2e-5, atol=1e-5), (scenario, audit_key)
        assert measured["live_env_physics_samples"] == original["sampled_physics_steps"]
        env_scores = []
        for env in range(32):
            mask = phase_masks["measured_after2s"][:, env]
            env_scores.append(float(np.sqrt(np.mean(applied[mask, env].astype(float) ** 2))) if mask.any() else np.nan)
            environment_rows.append({"scenario": scenario, "env_id": env, **report["per_environment"][env],
                                     "after2s_rms_all_joints_nm": env_scores[-1]})
        survivors = [env for env in range(32) if report["per_environment"][env]["survived"] and np.isfinite(env_scores[env])]
        representative = min(survivors, key=lambda env: abs(env_scores[env] - np.median([env_scores[i] for i in survivors])))
        gain_info = {key: {"minimum": float(trace[key].min()), "maximum": float(trace[key].max())} for key in (
            "physx_stiffness_nm_rad", "physx_damping_nm_s_rad", "physx_effort_limits_nm",
            "isaaclab_stiffness_nm_rad", "isaaclab_damping_nm_s_rad")}
        gain_info["measured_vs_nominal_postguard_max_abs_nm"] = float(np.abs(computed - nominal_post).max())
        gain_info["physx_vs_isaaclab_stiffness_max_abs"] = float(np.abs(trace["physx_stiffness_nm_rad"] - trace["isaaclab_stiffness_nm_rad"]).max())
        gain_info["physx_vs_isaaclab_damping_max_abs"] = float(np.abs(trace["physx_damping_nm_s_rad"] - trace["isaaclab_damping_nm_s_rad"]).max())
        result["scenarios"][scenario] = {"metadata": report["metadata"], "survived_envs": report["survived_envs"],
            "mean_forward_speed_m_s": report["mean_forward_speed_m_s"], "mean_directed_progress_m": report["mean_directed_progress_m"],
            "representative_env_id": representative, "representative_selection": "survivor closest to median 12-joint RMS over measured >=2s",
            "gains": gain_info, "phases": phases, "stairs": report.get("stairs"), "source_trace": trace_path.name}
        plot_alive = np.where(preroll[:, None], trace["first_episode_alive"], alive)
        masked_a = np.where(plot_alive[..., None], applied, np.nan)
        low, high = np.nanquantile(masked_a, [.05, .95], axis=1)
        fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
        for j, ax in enumerate(axes.flat):
            x = np.where(plot_alive[:, representative], applied[:, representative, j], np.nan)
            c = np.where(plot_alive[:, representative], computed[:, representative, j], np.nan)
            ax.fill_between(t, low[:, j], high[:, j], color="#3975b5", alpha=.18, label="32 env live p05-p95")
            ax.plot(t, c, lw=.8, color="#d17a20", ls="--", label="Computed estimate")
            ax.plot(t, x, lw=.8, color="#185c9c", label=f"Applied estimate env {representative}")
            for sign in (-1, 1):
                ax.axhline(sign * limits[j], color="#bf3535", lw=.6, ls=":")
            ax.axvspan(t.min(), 0, color="gray", alpha=.15)
            ax.axvspan(0, 2, color="#ddaa33", alpha=.09)
            ax.set_title(f"{names[j]}  (limit {limits[j]:g} Nm)", fontsize=10)
            ax.set_ylim(-limits[j] * 1.1, limits[j] * 1.1)
            ax.grid(alpha=.2)
            if j % 3 == 0:
                ax.set_ylabel("Estimated torque [Nm]")
            if j >= 9:
                ax.set_xlabel("Time from measurement start [s]")
        axes.flat[0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"V15 model_11699 / {scenario} / seed42 / 200Hz\nImplicit PD estimate; actual solver motor torque not measured. Gray=preroll, yellow=measured first 2s", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        save_fig(fig, figures / f"{scenario}_12joint_timeseries")
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        rows = [r for r in per_env_rows if r["scenario"] == scenario and r["phase"] == "measured_after2s"]
        for ax, field, label in zip(axes, ("rms_nm", "p99_nm", "fraction_ge80_limit"), ("RMS / limit", "p99 abs torque / limit", "Time fraction >= 80% limit")):
            distributions = [[r[field] / limits[j] if field != "fraction_ge80_limit" else r[field] for r in rows if r["joint"] == name] for j, name in enumerate(names)]
            ax.boxplot(distributions, whis=(5, 95), showfliers=True)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.25)
            ax.axhline(1, color="#bf3535", ls=":", lw=.8)
        axes[-1].set_xticks(np.arange(1, 13), names, rotation=35)
        fig.suptitle(f"V15 / {scenario} / 32 environments: valid samples at measured t >= 2s\nBox=env quartiles, whiskers=p05-p95; includes terminated env until first termination")
        fig.tight_layout(rect=(0, 0, 1, .94))
        save_fig(fig, figures / f"{scenario}_32env_distribution")
        for i, time in enumerate(t):
            for j, name in enumerate(names):
                series_rows.append({"scenario": scenario, "time_from_measurement_s": float(time), "env_id": representative,
                    "joint": name, "preroll": bool(preroll[i]), "alive": bool(plot_alive[i, representative]),
                    "applied_estimate_nm": float(applied[i, representative, j]), "computed_estimate_nm": float(computed[i, representative, j]),
                    "joint_velocity_rad_s": float(velocity[i, representative, j]), "live_env_p05_nm": float(low[i, j]),
                    "live_env_p95_nm": float(high[i, j])})
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    names = trace["joint_names"].tolist()
    x = np.arange(12)
    for offset, scenario in enumerate(SCENARIOS):
        values = result["scenarios"][scenario]["phases"]["measured_after2s"]
        for ax, field, label in zip(axes, ("rms_limit_fraction", "mean_abs_mechanical_power_w", "p99_torque_slew_nm_s"), ("Torque RMS / configured limit", "Mean |torque x velocity| [W/joint]", "p99 |d torque / dt| [Nm/s]")):
            ax.bar(x + (offset - 1) * .25, values[field], width=.24, label=scenario)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    axes[-1].set_xticks(x, names, rotation=35)
    fig.suptitle("V15 model_11699 / alive samples after measured first 2s\nImplicit PD estimates; mechanical power is not motor thermal dissipation")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_fig(fig, figures / "terrain_power_slew_comparison")
    result["data_files"] = files
    result["verification"] = {"trace_dimensions_and_finite": True, "original_actuation_metrics_reproduced": True,
                                "original_actor_checkpoint_hash_equal": len({x["metadata"]["actor_sha256"] for x in result["scenarios"].values()}) == 1}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_csv(output / "per_joint_segments.csv", per_joint_rows)
    write_csv(output / "per_environment_joint_segments.csv", per_env_rows)
    write_csv(output / "representative_timeseries.csv", series_rows)
    write_csv(output / "environment_outcomes.csv", environment_rows)
    print(json.dumps({scenario: {key: result["scenarios"][scenario]["phases"]["measured_after2s"][key] for key in (
        "pooled_p99_applied_nm", "pooled_saturation_fraction", "pooled_above80_fraction", "pooled_guard_clip_fraction",
        "pooled_preguard_over_limit_fraction", "pooled_gain_corner_counterfactual_over_limit_fraction",
        "pooled_mean_abs_joint_power_w", "pooled_p99_slew_nm_s")} for scenario in SCENARIOS}, indent=2))


if __name__ == "__main__":
    main()
