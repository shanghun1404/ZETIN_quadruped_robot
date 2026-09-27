"""Summarize V15 training logs and evaluations against docs/v15_design.md gates.

    python summarize.py <results_dir> <output.json> [slurm-<job>.out ...]
"""

import json
import re
import sys
from pathlib import Path

LOG_FIELDS = {
    "mean_reward": r"Mean reward:\s+(-?[\d.]+)",
    "mean_episode_length": r"Mean episode length:\s+([\d.]+)",
    "terrain_level": r"Curriculum/terrain_levels:\s+([\d.]+)",
    "iteration_time_s": r"Iteration time:\s+([\d.]+)s",
}


def training_progress(log_path):
    """First/last values of the RSL-RL console summary, per logged field."""
    text = Path(log_path).read_text(errors="replace")
    iterations = [int(i) for i in re.findall(r"Learning iteration (\d+)/", text)]
    result = {"log": Path(log_path).name, "iterations_logged": len(iterations),
              "last_iteration": iterations[-1] if iterations else None}
    for name, pattern in LOG_FIELDS.items():
        values = [float(v) for v in re.findall(pattern, text)]
        if values:
            result[name] = {"first": values[0], "last": values[-1], "max": max(values)}
    return result


def locomotion_row(path, data):
    path_summary = data.get("path") or {}
    return {"file": path.name, "scenario": data["metadata"]["scenario"], "seed": data["metadata"]["seed"],
            "heading_control": data["metadata"]["heading_control"], "survived": data["survived_envs"],
            "initial": data["initial_envs"], "progress_m": data["mean_directed_progress_m"],
            "directed_stair_success": path_summary.get("directed_stair_success_count"),
            "over_limit_request_fraction": data["actuation"]["over_limit_request_fraction"],
            "applied_saturation_fraction": data["actuation"]["applied_saturation_fraction"],
            "actor_sha256": data["metadata"]["actor_sha256"]}


def recovery_row(path, data):
    return {"file": path.name, "start": data["start"], "seed": data["metadata"]["seed"],
            "success": data["success_count"], "num_envs": data["num_envs"],
            "success_rate": data["success_rate"], "median_time_to_recover_s": data["median_time_to_recover_s"],
            "checkpoint_sha256": data["metadata"]["checkpoint_sha256"]}


def main(results_dir, output, *logs):
    locomotion, recovery = [], []
    for path in sorted(Path(results_dir).rglob("*.json")):
        data = json.loads(path.read_text())
        if "actuation" in data:
            locomotion.append(locomotion_row(path, data))
        elif "success_rate" in data:
            recovery.append(recovery_row(path, data))
    summary = {"training": [training_progress(log) for log in logs],
               "locomotion": locomotion, "recovery": recovery}
    Path(output).write_text(json.dumps(summary, indent=2) + "\n")
    for row in summary["training"]:
        print(json.dumps(row))
    for row in locomotion:
        print(f"{row['file']}: survived {row['survived']}/{row['initial']}, progress {row['progress_m']:.2f} m, "
              f"directed stairs {row['directed_stair_success']}, over-limit {row['over_limit_request_fraction']:.3f}")
    for row in recovery:
        print(f"{row['file']}: {row['start']} success {row['success']}/{row['num_envs']}, "
              f"median recovery {row['median_time_to_recover_s']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
