"""Plot saved V16 results without launching a physics simulator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(job: Path):
    figures = job / "figures"
    figures.mkdir(exist_ok=True)
    manifest_path = job / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    checkpoint = job / "checkpoint.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        rows = state["history"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        generation = [r["generation"] for r in rows]
        axes[0].plot(generation, [r["score_max"] for r in rows], label="best in population")
        axes[0].plot(generation, [r["score_mean"] for r in rows], label="population mean")
        axes[0].set_ylabel("Training score")
        axes[0].legend()
        axes[1].plot(generation, [r["survived"] for r in rows])
        axes[1].set_ylabel("Surviving candidates")
        axes[2].plot(generation, [r["progress_best_m"] for r in rows])
        axes[2].set_ylabel("Best progress [m]")
        axes[2].set_xlabel("CEM generation")
        fig.suptitle("V16 controller parameter training (holdout is separate)")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"training.{suffix}", dpi=150)
        plt.close(fig)
    for trace_path in sorted(job.glob("trace_seed*.npz")):
        data = np.load(trace_path, allow_pickle=False)
        times = data["time_s"]
        alive = data["alive"]
        n = alive.shape[1]
        summaries = json.loads(trace_path.with_name(trace_path.name.replace("trace_", "eval_").replace(".npz", ".json")).read_text())
        forward_axis = 1 if summaries.get("walk_axis_world", "x") == "y" else 0
        survivors = np.flatnonzero(alive[-1])
        eligible = survivors if len(survivors) else np.arange(n)
        progress = np.asarray(summaries["progress_m"])
        env = int(eligible[np.argsort(progress[eligible])[len(eligible) // 2]])
        commands = data["torque_command_nm"][:, env]
        requests = data["torque_requested_nm"][:, env]
        joint_names = data["joint_names"].tolist()
        limits = np.array([24.8 if name.endswith("HAA") else 48. for name in joint_names])
        failed = np.flatnonzero(~alive[:, env])
        failure_time = float(times[failed[0]]) if len(failed) else None
        valid = alive[:, env] & (times >= 1.)
        if not valid.any():
            valid = alive[:, env]
        segment = commands[valid]
        statistics = {
            "trace": trace_path.name, "representative_env": env,
            "representative_selection": "median final progress among survivors; all environments if none survive",
            "survived_envs": int(alive[-1].sum()), "num_envs": n,
            "failure_time_s": failure_time, "sample_interval_s": float(np.median(np.diff(times))),
            "effort_kind": "direct-effort command, not measured solver contact/friction wrench",
            "joint_order": joint_names,
            "peak_abs_command_nm": np.max(np.abs(segment), axis=0).tolist() if len(segment) else None,
            "rms_command_nm": np.sqrt(np.mean(segment**2, axis=0)).tolist() if len(segment) else None,
            "p99_abs_command_nm": np.percentile(np.abs(segment), 99, axis=0).tolist() if len(segment) else None,
        }
        (figures / f"{trace_path.stem}_metrics.json").write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
        fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(times, requests[:, j], color="0.6", linewidth=.6, label="request")
            ax.plot(times, commands[:, j], linewidth=.8, label="command")
            ax.axhline(limits[j], color="r", linestyle="--", linewidth=.5)
            ax.axhline(-limits[j], color="r", linestyle="--", linewidth=.5)
            ax.set_title(joint_names[j], fontsize=10)
            if failure_time is not None:
                ax.axvline(failure_time, color="orange", linestyle=":")
            if j % 3 == 0:
                ax.set_ylabel("Torque [Nm]")
            if j >= 9:
                ax.set_xlabel("Time [s]")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"V16 direct effort, environment {env}; survived {int(alive[-1].sum())}/{n}")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{trace_path.stem}_torque.{suffix}", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        root = data["root_position_m"][:, env]
        for j, label in enumerate(("x", "y", "z")):
            axes[0].plot(times, root[:, j], label=label)
        axes[0].set_ylabel("Base actor position [m]")
        axes[0].legend()
        for i in range(n):
            axes[1].plot(times, data["root_position_m"][:, i, forward_axis], alpha=.3)
        axes[1].set_ylabel(f"All base {'xy'[forward_axis]} [m]")
        axes[2].plot(times, alive.sum(axis=1), label="surviving environments")
        axes[2].set_ylabel("Survivors")
        axes[2].set_xlabel("Time [s]")
        for ax in axes:
            if failure_time is not None:
                ax.axvline(failure_time, color="orange", linestyle=":")
        fig.suptitle("Absorbing failure: post-failure poses may be administrative resets")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{trace_path.stem}_body.{suffix}", dpi=150)
        plt.close(fig)
        if "whole_com_position_m" in data and "scheduled_stance" in data:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(times, data["scheduled_stance"][:, env].sum(axis=1), label="scheduled stance")
            proxy_contact = data["contact_force_w"][:, env, :, 2] > 20.
            axes[0].plot(times, proxy_contact.sum(axis=1), label="foot net normal force >20 N", alpha=.8)
            axes[0].set_ylabel("Foot count")
            axes[0].legend(fontsize=8)
            for axis, label in enumerate(("x", "y")):
                axes[1].plot(times, data["whole_com_position_m"][:, env, axis], label=f"whole COM {label}")
                if "whole_com_target_xy_m" in data and summaries.get("gait_mode") == "prepared_crawl":
                    axes[1].plot(times, data["whole_com_target_xy_m"][:, env, axis], linestyle="--", label=f"target {label}")
                if "whole_com_reference_xy_m" in data and summaries.get("smooth_com_reference"):
                    axes[1].plot(times, data["whole_com_reference_xy_m"][:, env, axis], linestyle=":", label=f"smooth reference {label}")
            axes[1].set_ylabel("Whole COM / target [m]")
            axes[1].legend(fontsize=8)
            if "support_triangle_margin_m" in data and summaries.get("gait_mode") == "prepared_crawl":
                axes[2].plot(times, data["support_triangle_margin_m"][:, env], label="next 3-foot support triangle margin [m]")
                margin_gate = manifest.get("prepared_crawl_gates", {}).get("liftoff_whole_com_margin_m", .02)
                axes[2].axhline(margin_gate, linestyle="--", color="0.5", label="liftoff margin gate")
                axes[2].set_ylabel("Signed margin [m]")
            elif "stance_foot_speed_m_s" in data:
                axes[2].plot(times, data["stance_foot_speed_m_s"][:, env], label="stance point speed [m/s]")
                axes[2].set_ylabel("Stance point speed [m/s]")
            axes[2].legend(fontsize=8)
            axes[2].set_xlabel("Time [s]")
            for ax in axes:
                if failure_time is not None:
                    ax.axvline(failure_time, color="orange", linestyle=":")
            fig.suptitle(f"V16 contact / whole COM diagnostics, environment {env}; foot points are approximations")
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(figures / f"{trace_path.stem}_support.{suffix}", dpi=150)
            plt.close(fig)
        if "force_qp_constraint_violation" in data and summaries.get("force_allocator") in ("qp", "wbc"):
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            control_times = data["control_time_s"]
            if "whole_com_reference_velocity_m_s" in data:
                axes[0].plot(control_times, np.linalg.norm(data["whole_com_reference_velocity_m_s"][:, env], axis=1), label="COM reference speed")
                axes[0].axhline(.30, color="0.5", linestyle="--", label="reference limit")
            axes[0].plot(times, np.linalg.norm(data["root_com_velocity_w"][:, env, :2], axis=1), label="actual root COM speed")
            axes[0].set_ylabel("Horizontal speed [m/s]")
            axes[0].legend(fontsize=8)
            if "whole_com_reference_acceleration_m_s2" in data:
                axes[1].plot(control_times, np.linalg.norm(data["whole_com_reference_acceleration_m_s2"][:, env], axis=1), label="COM reference acceleration")
                axes[1].axhline(1.2, color="0.5", linestyle="--", label="reference limit")
            axes[1].plot(control_times, np.linalg.norm(data["projected_base_acceleration"][:, env, :2], axis=1), label="optimized / projected root acceleration")
            axes[1].set_ylabel("Horizontal acceleration [m/s²]")
            axes[1].legend(fontsize=8)
            violation = data["force_qp_constraint_violation"][:, env]
            axes[2].semilogy(control_times, np.maximum(violation, 1e-8), label="returned QP constraint violation")
            axes[2].axhline(1e-3, color="0.5", linestyle="--", label="numerical feasible tolerance")
            axes[2].set_ylabel("Normalized violation")
            axes[2].set_xlabel("Time [s]")
            axes[2].legend(fontsize=8)
            fig.suptitle(f"V16 {summaries['force_allocator']} diagnostics, environment {env}; reference bounds do not bound actual motion")
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(figures / f"{trace_path.stem}_control.{suffix}", dpi=150)
            plt.close(fig)
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    print(analyze(parser.parse_args().job))
