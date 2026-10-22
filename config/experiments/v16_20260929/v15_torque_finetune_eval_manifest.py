"""Bind the completed536 checkpoint and fixed339 reference before GPU evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

ACTOR339 = Path("/home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt")
TRAINING_OVERLAY_SHA256 = "3b3e003e44ab542c7796c55e8bf1f1a9e68b3d29d86e58af554b07189b280aeb"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bind_inputs(train_root, template, actor339=ACTOR339):
    markers = list(Path(train_root).glob("guide_v15_torque_finetune/*/run_completed.json"))
    require(len(markers) == 1, "Exactly one completed536 run required")
    marker = markers[0]
    done = json.loads(marker.read_text())
    require(done.get("training_completed") is True and done.get("nonfinite_steps") == 0, "Training did not complete with finite rollout diagnostics")
    require(done["last_learning_iteration"] == 11949, "Expected536 final iteration11949")
    require(len(done["policy_std_min_max"]) == 2 and all(math.isfinite(v) and v > 0 for v in done["policy_std_min_max"]), "Invalid saved policy std")
    checkpoint = Path(done["checkpoint"]).resolve()
    require(checkpoint.parent == marker.parent.resolve() and checkpoint.name == "model_11949.pt", "Completed checkpoint location/name mismatch")
    require(digest(checkpoint) == done["checkpoint_sha256"], "Completed checkpoint hash mismatch")
    require(digest(actor339) == template["baseline"]["actor_sha256"], "Frozen339 reference hash mismatch")
    contract_path = marker.with_name("finetune_actual_contract.json")
    require(digest(contract_path) == done["actual_contract_sha256"], "Actual training contract hash mismatch")
    contract = json.loads(contract_path.read_text())
    expected = {"algorithm_optimizer_lr_match": True, "schedule_after": "fixed", "actual_guard_fraction": .97,
                "actual_cutoff_hz": 23., "starting_learning_iteration": 11700, "num_learning_iterations": 250,
                "num_envs": 512, "seed": 47, "rollout_steps_per_env": 24, "episode_length_s": 20.,
                "physics_dt_s": .005, "control_dt_s": .02, "max_delay_steps": 3, "physics_diverged_termination": True}
    require(all(contract.get(key) == value for key, value in expected.items()), "536 actual training settings differ from reviewed pilot")
    require(math.isclose(contract["algorithm_learning_rate_after"], 3.375e-5, rel_tol=0, abs_tol=1e-15), "Algorithm LR mismatch")
    groups = contract["optimizer_group_lr_after"]
    require(groups and all(math.isclose(v, 3.375e-5, rel_tol=0, abs_tol=1e-15) for v in groups), "Optimizer group LR mismatch")
    require(math.isclose(contract["actual_lpf_alpha"], .005 / (.005 + 1 / (2 * math.pi * 23)), rel_tol=1e-12), "Actual training LPF alpha mismatch")
    training_manifest = marker.with_name("v15_manifest.json")
    profile = json.loads(training_manifest.read_text())
    require(profile["initial_checkpoint_sha256"] == template["baseline"]["actor_sha256"] and profile["resume"] is True, "536 provenance is not frozen339 resume")
    require(profile["action_lpf_hz"] == 23. and profile["effort_guard_fraction"] == .97, "536 training actuation profile mismatch")
    require(profile["overlay_sha256"] == TRAINING_OVERLAY_SHA256 and digest(Path(train_root) / "v15_torque_finetune.py") == TRAINING_OVERLAY_SHA256, "Reviewed536 overlay source hash mismatch")
    manifest = copy.deepcopy(template)
    before, after = manifest["candidates"]
    require(before["label"] == "pre339_g097_lpf23" and after["label"] == "adapt536_g097_lpf23", "Unexpected candidate labels/order")
    before["actor_sha256"] = digest(actor339)
    after["actor_sha256"] = digest(checkpoint)
    manifest["training536_inputs"] = {"checkpoint_sha256": after["actor_sha256"], "run_completed_sha256": digest(marker),
                                      "actual_contract_sha256": digest(contract_path), "training_manifest_sha256": digest(training_manifest),
                                      "training_overlay_sha256": TRAINING_OVERLAY_SHA256}
    paths = {before["label"]: str(Path(actor339).resolve()), after["label"]: str(checkpoint)}
    return {"manifest": manifest, "model_paths": paths, "actual_training_contract": contract,
            "copy_inputs": {"training536_run_completed.json": marker,
                            "training536_actual_contract.json": contract_path,
                            "training536_manifest.json": training_manifest}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = bind_inputs(args.training_dir, json.loads(args.template.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, key in (("v15_torque_finetune_eval_manifest.json", "manifest"), ("v15_torque_finetune_eval_models.json", "model_paths")):
        (args.output_dir / name).write_text(json.dumps(result[key], indent=2) + "\n")
    for name, path in result["copy_inputs"].items():
        shutil.copy2(path, args.output_dir / name)
    shutil.copy2(args.training_dir / "source_sha256.txt", args.output_dir / "training536_source_sha256.txt")
    print(json.dumps({"bound_models": result["manifest"]["candidates"], "training536_verified": True}))


if __name__ == "__main__":
    main()
