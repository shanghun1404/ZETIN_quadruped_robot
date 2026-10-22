"""Bind the completed torque reward pilot checkpoint and fixed339 reference before GPU evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

ACTOR339 = Path("/home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt")
TRAINING_OVERLAY_SHA256 = "87db5a6165e5017cb3e1c684ddb905d0ff835b8980c4c2320c72843f5e9f88e4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bind_inputs(train_root, template, actor339=ACTOR339):
    markers = list(Path(train_root).glob("guide_v15_torque_reward_finetune/*/run_completed.json"))
    require(len(markers) == 1, "Exactly one completed torque reward pilot run required")
    marker = markers[0]
    done = json.loads(marker.read_text())
    require(done.get("training_completed") is True and done.get("nonfinite_steps") == 0, "Training did not complete with finite rollout diagnostics")
    require(done["last_learning_iteration"] == 11949, "Expectedreward pilot final iteration11949")
    require(len(done["policy_std_min_max"]) == 2 and all(math.isfinite(v) and v > 0 for v in done["policy_std_min_max"]), "Invalid saved policy std")
    checkpoint = Path(done["checkpoint"]).resolve()
    require(checkpoint.parent == marker.parent.resolve() and checkpoint.name == "model_11949.pt", "Completed checkpoint location/name mismatch")
    require(digest(checkpoint) == done["checkpoint_sha256"], "Completed checkpoint hash mismatch")
    require(digest(actor339) == template["baseline"]["actor_sha256"], "Frozen339 reference hash mismatch")
    contract_path = marker.with_name("finetune_actual_contract.json")
    require(digest(contract_path) == done["actual_contract_sha256"], "Actual training contract hash mismatch")
    contract = json.loads(contract_path.read_text())
    expected = {"algorithm_optimizer_lr_match": True, "schedule_after": "fixed", "actual_guard_fraction": .99,
                "actual_cutoff_hz": 25., "starting_learning_iteration": 11700, "num_learning_iterations": 250,
                "num_envs": 512, "seed": 47, "rollout_steps_per_env": 24, "episode_length_s": 20.,
                "physics_dt_s": .005, "control_dt_s": .02, "max_delay_steps": 3, "physics_diverged_termination": True, "actual_joint_torques_weight": -.000625,
                "actual_joint_torques_function": "isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.rewards.joint_torques_penalty"}
    require(all(contract.get(key) == value for key, value in expected.items()), "reward pilot actual training settings differ from reviewed pilot")
    require(math.isclose(contract["algorithm_learning_rate_after"], 3.375e-5, rel_tol=0, abs_tol=1e-15), "Algorithm LR mismatch")
    groups = contract["optimizer_group_lr_after"]
    require(groups and all(math.isclose(v, 3.375e-5, rel_tol=0, abs_tol=1e-15) for v in groups), "Optimizer group LR mismatch")
    require(math.isclose(contract["actual_lpf_alpha"], .005 / (.005 + 1 / (2 * math.pi * 25)), rel_tol=1e-12), "Actual training LPF alpha mismatch")
    rewards = contract["torque_reward_contract"]
    require(rewards["torque_reward_multiplier"] == 1.25 and rewards["joint_torques_weight_before"] == -.0005 and rewards["joint_torques_weight_after"] == -.000625, "Torque reward multiplier/weights mismatch")
    before_weights, after_weights = rewards["weights_before"], rewards["weights_after"]
    require(before_weights.keys() == after_weights.keys() and before_weights["joint_torques"] == -.0005 and after_weights["joint_torques"] == -.000625, "Reward inventory/torque weight mismatch")
    require(all(after_weights[name] == value for name, value in before_weights.items() if name != "joint_torques"), "Other reward changed")
    require(contract["actual_reward_weights"] == {name: value for name, value in after_weights.items() if value != 0}, "Actual reward manager weights mismatch")
    training_manifest = marker.with_name("v15_manifest.json")
    profile = json.loads(training_manifest.read_text())
    require(profile["initial_checkpoint_sha256"] == template["baseline"]["actor_sha256"] and profile["resume"] is True, "reward pilot provenance is not frozen339 resume")
    require(profile["reward_changes"] is True and profile["torque_reward_contract"] == rewards, "Reward manifest/actual contract mismatch")
    require(profile["original_trainer_sha256"] == "3da8cdcefa600f851795d3d48cdd30a6172b55070556ff8d888971057a1391d2", "Original trainer mismatch")
    require(profile["action_lpf_hz"] == 25. and profile["effort_guard_fraction"] == .99, "reward pilot training actuation profile mismatch")
    require(profile["overlay_sha256"] == TRAINING_OVERLAY_SHA256 and digest(Path(train_root) / "v15_torque_reward_finetune.py") == TRAINING_OVERLAY_SHA256, "Reviewedreward pilot overlay source hash mismatch")
    manifest = copy.deepcopy(template)
    require(len(manifest["candidates"]) == 1, "Only the new torque reward actor is evaluated")
    after = manifest["candidates"][0]
    require(after["label"] == "adapt_torque1p25_g099_lpf25", "Unexpected candidate label")
    after["actor_sha256"] = digest(checkpoint)
    manifest["reward_training_inputs"] = {"checkpoint_sha256": after["actor_sha256"], "run_completed_sha256": digest(marker),
                                      "actual_contract_sha256": digest(contract_path), "training_manifest_sha256": digest(training_manifest),
                                      "training_overlay_sha256": TRAINING_OVERLAY_SHA256}
    manifest.setdefault("comparison_design", {})["evaluation_submission_status"] = "Completed training inputs bound; physical evaluation pending"
    manifest["comparison_design"]["training_root"] = str(Path(train_root).resolve())
    paths = {after["label"]: str(checkpoint)}
    return {"manifest": manifest, "model_paths": paths, "actual_training_contract": contract,
            "copy_inputs": {"reward_training_run_completed.json": marker,
                            "reward_training_actual_contract.json": contract_path,
                            "reward_training_manifest.json": training_manifest}}


def verify_baseline_reuse(template):
    reuse = template["baseline_reuse"]
    require(reuse["job"] == 508 and reuse["fresh_reference_physics_run"] is False, "Expected original508 reuse")
    verified = {}
    for group in ("artifact_sha256", "immutable_source_sha256"):
        for path, expected in reuse[group].items():
            require(digest(path) == expected, f"508 baseline reuse hash mismatch: {path}")
            verified[path] = expected
    require(len(reuse["artifact_sha256"]) == 8, "Expected flat/rough four raw artifacts each")
    return {"baseline_job": 508, "fresh_reference_physics_run": False, "verified_sha256": verified,
            "metadata_by_scenario": reuse["metadata_by_scenario"], "physical_contract_verified_from_original_traces": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text())
    reuse = verify_baseline_reuse(template)
    result = bind_inputs(args.training_dir, template)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v15_torque_reward_eval_baseline_reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    for name, key in (("v15_torque_reward_eval_manifest.json", "manifest"), ("v15_torque_reward_eval_models.json", "model_paths")):
        (args.output_dir / name).write_text(json.dumps(result[key], indent=2) + "\n")
    for name, path in result["copy_inputs"].items():
        shutil.copy2(path, args.output_dir / name)
    shutil.copy2(args.training_dir / "source_sha256.txt", args.output_dir / "reward_training_source_sha256.txt")
    print(json.dumps({"bound_models": result["manifest"]["candidates"], "reward_training_verified": True}))


if __name__ == "__main__":
    main()
