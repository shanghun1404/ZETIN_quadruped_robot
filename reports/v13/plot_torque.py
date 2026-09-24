"""Plot the saved V12/V13 flat-ground torque comparison without Isaac Lab."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COLORS = {"v12": "#31688e", "v13": "#d95f02"}
SEEDS = (42, 43, 44)


def load_reports() -> dict[str, list[dict]]:
    reports = {
        variant: [json.loads((DATA / f"{variant}_flat_seed{seed}.json").read_text()) for seed in SEEDS]
        for variant in ("v12", "v13")
    }
    reference = reports["v12"][0]
    for variant, runs in reports.items():
        for seed, run in zip(SEEDS, runs, strict=True):
            assert run["variant"] == variant
            assert run["metadata"]["seed"] == seed
            assert run["joint_order"] == reference["joint_order"]
            assert run["metadata"]["student_sha256"] == reference["metadata"]["student_sha256"]
            assert run["metadata"]["usd_sha256"] == reference["metadata"]["usd_sha256"]
            assert run["metadata"]["physics_dt_s"] == 0.005
            assert run["metadata"]["measured_s"] == 12.0
    return reports


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(ROOT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(ROOT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary(reports: dict[str, list[dict]]) -> None:
    metrics = (
        ("Requested torque P99", "Nm", "p99_requested_nm", 1.0),
        ("Requests over motor limit", "% of joint samples", "over_limit_request_fraction", 100.0),
        ("Applied torque saturation", "% of joint samples", "applied_saturation_fraction", 100.0),
        ("Applied torque slew P99", "kNm/s", "p99_applied_torque_slew_nm_s", 0.001),
        ("Mean absolute joint power", "W", "mean_joint_power_w", 1.0),
        ("Forward speed", "m/s", "mean_forward_speed_m_s", 1.0),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for ax, (title, unit, key, scale) in zip(axes.flat, metrics, strict=True):
        values = {}
        for variant in ("v12", "v13"):
            values[variant] = np.asarray(
                [run["actuation"][key] if key in run["actuation"] else run[key] for run in reports[variant]]
            ) * scale
        means = [float(values[variant].mean()) for variant in ("v12", "v13")]
        for index, variant in enumerate(("v12", "v13")):
            samples = values[variant]
            mean = means[index]
            ax.bar(index, mean, width=0.55, color=COLORS[variant], alpha=0.82)
            ax.errorbar(
                index, mean, yerr=[[mean - samples.min()], [samples.max() - mean]],
                color="#222222", capsize=5, fmt="none", lw=1.1,
            )
            ax.scatter(index + np.linspace(-0.08, 0.08, len(samples)), samples, s=17,
                       color="#222222", zorder=4)
            ax.text(index, mean + max(values["v12"].max(), values["v13"].max()) * 0.04,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=10)
        change = 100.0 * (means[1] / means[0] - 1.0)
        ax.text(0.97, 0.96, f"{change:+.1f}%", transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=COLORS["v13"], fontweight="bold")
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_ylabel(unit)
        ax.set_xticks((0, 1), ("V12", "V13"))
        ax.set_ylim(0, max(values["v12"].max(), values["v13"].max()) * 1.30)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("V12 vs V13: flat-ground actuation", fontsize=17, y=0.99)
    fig.text(0.5, 0.945, "Three paired seeds, 32 environments per seed, 12 s at 200 Hz; forward command +0.4 m/s",
             ha="center", fontsize=10)
    fig.text(0.5, 0.005,
             "Requested torque is computed before motor clipping. Motor limits: HAA 24.8 Nm; HFE/KFE 48 Nm."
             " Bars show seed means, whiskers min-max, dots individual seeds.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.92), h_pad=2.4, w_pad=2.0)
    save_figure(fig, "torque_summary")


def load_trace(variant: str) -> dict[str, np.ndarray]:
    path = DATA / f"{variant}_trace_seed42.npz"
    with np.load(path, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    assert str(trace["variant"]) == variant
    assert int(trace["seed"]) == 42
    assert trace["computed_torque_nm"].shape == trace["applied_torque_nm"].shape == (2400, 32, 12)
    assert trace["alive"].shape == (2400, 32)
    assert trace["time_s"].shape == (2400,)
    return trace


def plot_trace() -> None:
    traces = {variant: load_trace(variant) for variant in ("v12", "v13")}
    assert np.array_equal(traces["v12"]["time_s"], traces["v13"]["time_s"])
    assert np.array_equal(traces["v12"]["joint_names"], traces["v13"]["joint_names"])
    time_s = traces["v12"]["time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.8))
    for variant, trace in traces.items():
        requested = np.abs(trace["computed_torque_nm"])
        live_requested = np.where(trace["alive"][..., None], requested, np.nan)
        p99 = np.nanquantile(live_requested, 0.99, axis=(1, 2))
        peak = np.nanmax(live_requested, axis=(1, 2))
        axes[0, 0].plot(time_s, p99, color=COLORS[variant], lw=0.7, alpha=0.8, label=variant.upper())
        axes[0, 1].plot(time_s, peak, color=COLORS[variant], lw=0.7, alpha=0.8, label=variant.upper())
    axes[0, 0].set_title("Requested torque: P99 across live motors at each physics step")
    axes[0, 1].set_title("Requested torque: maximum across live motors at each physics step")
    for ax in axes[0]:
        ax.set_xlabel("Time after warm-up (s)")
        ax.set_ylabel("Absolute requested torque (Nm)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 12)
        ax.set_ylim(bottom=0)

    v12_request = traces["v12"]["computed_torque_nm"]
    peak_step, env_id, joint_id = np.unravel_index(np.abs(v12_request).argmax(), v12_request.shape)
    window_steps = 401
    lower = max(0, min(peak_step - window_steps // 2, len(time_s) - window_steps))
    upper = lower + window_steps
    joint_name = str(traces["v12"]["joint_names"][joint_id])
    limit = 24.8 if joint_name.endswith("HAA") else 48.0
    joint_max = max(
        float(np.abs(traces[variant]["computed_torque_nm"][lower:upper, env_id, joint_id]).max())
        for variant in ("v12", "v13")
    )
    for ax, variant in zip(axes[1], ("v12", "v13"), strict=True):
        trace = traces[variant]
        ax.plot(time_s[lower:upper], trace["computed_torque_nm"][lower:upper, env_id, joint_id],
                color=COLORS[variant], lw=0.95, label="Requested before clipping")
        ax.plot(time_s[lower:upper], trace["applied_torque_nm"][lower:upper, env_id, joint_id],
                color="#202020", lw=0.85, alpha=0.85, label="Applied to motor")
        ax.axhline(limit, color="#777777", ls="--", lw=0.9)
        ax.axhline(-limit, color="#777777", ls="--", lw=0.9, label=f"Motor limit ±{limit:g} Nm")
        ax.set_title(f"{variant.upper()}: {joint_name}, environment {env_id}")
        ax.set_xlabel("Time after warm-up (s)")
        ax.set_ylabel("Signed torque (Nm)")
        ax.set_ylim(-joint_max * 1.12, joint_max * 1.12)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("V12 vs V13: 200 Hz torque traces | seed 42", fontsize=17, y=0.99)
    fig.text(0.5, 0.005,
             "Bottom panels use the same joint/environment and a 2 s window around V12's largest requested spike."
             " This is an extreme example; the top panels include all live motors.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=2.4)
    save_figure(fig, "torque_trace_seed42")


if __name__ == "__main__":
    plot_summary(load_reports())
    if all((DATA / f"{variant}_trace_seed42.npz").is_file() for variant in ("v12", "v13")):
        plot_trace()
