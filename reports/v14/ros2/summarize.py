"""Summarize the one-robot ROS2 vs in-process A/B evaluations (results/ros2/*.json)."""

import json
import sys
from pathlib import Path


def row(path):
    data = json.loads(path.read_text())
    meta, actuation, env = data["metadata"], data["actuation"], data["per_environment"][0]
    parity = meta.get("ros2_parity") or {}
    stairs = data.get("stairs") or {}
    return {
        "file": path.name,
        "control": meta["control"],
        "scenario": meta["scenario"],
        "seed": meta["seed"],
        "survived": bool(env["survived"]),
        "first_termination_s": env["first_termination_s"],
        "first_termination_terms": env["first_termination_terms"],
        "directed_progress_m": env["directed_progress_m"],
        "stairs_traversed": stairs.get("traversed_count"),
        "applied_saturation_fraction": actuation["applied_saturation_fraction"],
        "over_limit_request_fraction": actuation["over_limit_request_fraction"],
        "mean_joint_power_w": actuation["mean_joint_power_w"],
        "parity_target_max_rad": (parity.get("policy_target") or {}).get("max_abs_rad"),
        "parity_command_max_rad": (parity.get("joint_command") or {}).get("max_abs_rad"),
        "parity_samples": (parity.get("joint_command") or {}).get("samples"),
        "actor_sha256": meta["actor_sha256"],
        "student_sha256": meta["student_sha256"],
        "usd_sha256": meta["usd_sha256"],
    }


def main(directory, output):
    rows = sorted((row(p) for p in Path(directory).glob("*.json")),
                  key=lambda r: (r["scenario"], r["seed"], r["control"], r["file"]))
    for key in ("actor_sha256", "student_sha256", "usd_sha256"):
        if len({r[key] for r in rows}) != 1:
            raise RuntimeError(f"evaluations disagree on {key}")
    Path(output).write_text(json.dumps(rows, indent=2) + "\n")
    print("| 지형 | 시드 | 제어 | 생존 | 종료(s) | 진행(m) | 계단 통과 | 포화 | 파워(W) | parity 최대(rad) |")
    print("|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        parity = r["parity_command_max_rad"]
        ended = r["first_termination_s"]
        print(f"| {r['scenario']} | {r['seed']} | {r['control']} | {'O' if r['survived'] else 'X'} "
              f"| {f'{ended:.2f}' if ended is not None else '—'} "
              f"| {r['directed_progress_m']:.2f} | {r['stairs_traversed'] if r['stairs_traversed'] is not None else '—'} "
              f"| {100 * r['applied_saturation_fraction']:.1f}% | {r['mean_joint_power_w']:.1f} "
              f"| {f'{parity:.1e}' if parity is not None else '—'} |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
