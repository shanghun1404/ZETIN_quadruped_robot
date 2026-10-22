#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=v15-torque-reward-holdout
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
set -euo pipefail
training_job=${1:?completed torque reward training job ID required}
[[ "$training_job" =~ ^[0-9]+$ ]] || exit 2
holdout_seed=${2:?holdout seed43 or44 required}
[[ "$holdout_seed" == 43 || "$holdout_seed" == 44 ]] || exit 2
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
audit_config=$v15/config/experiments/v16_20260929
audit_output=$v15/results/v16_20260929/v15_torque_reward_holdout_${SLURM_JOB_ID}
actor=$v15/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
mkdir -p "$audit_output"
cp "$audit_config/v15_torque_capture.py" "$audit_config/v15_torque_reward_eval_capture.py" "$audit_config/v15_torque_reward_eval_manifest.py" "$audit_config/v15_torque_reward_eval_manifest_template.json" "$audit_config/v15_torque_reward_holdout.sbatch" "$audit_config/v15_torque_reward_holdout_selection.json" "$audit_config/v15_torque_reward_screening_summary.json" "$audit_config/v15_torque_holdout_protocol.json" "$audit_output/"
/opt/isaacsim-env/bin/python "$audit_output/v15_torque_reward_eval_manifest.py" \
  --training-dir "$v15/results/v16_20260929/v15_torque_reward_finetune_${training_job}" \
  --template "$audit_output/v15_torque_reward_eval_manifest_template.json" --output-dir "$audit_output"
/opt/isaacsim-env/bin/python - "$audit_output" "$training_job" <<'PYSELECT'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
selection = json.loads((root / "v15_torque_reward_holdout_selection.json").read_text())
summary_path = root / "v15_torque_reward_screening_summary.json"
assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == selection["screening_summary_sha256"]
summary = json.loads(summary_path.read_text())
label = "adapt_torque1p25_g099_lpf25"
assert selection["selected_candidate"] == summary["screening_selected_candidate"] == label
assert selection["training_job"] == int(sys.argv[2])
assert selection["seed42_all_gates_pass"] is True and summary["all_candidates_complete"] is True
candidate = summary["candidates"][label]
assert candidate["all_gates_pass"] is True
assert set(candidate["comparisons"]) == {"flat", "rough"}
assert all(case["control_pass"] and case["torque_pass"] for case in candidate["comparisons"].values())
manifest = json.loads((root / "v15_torque_reward_eval_manifest.json").read_text())
assert summary["control_gate"] == manifest["screening"]["control_gate"]
assert summary["torque_gate"] == manifest["screening"]["torque_gate"]
assert candidate["config"] == selection["config"] == manifest["candidates"][0]
assert selection["source_eval_dir"] == "/home/shanghun1404/v15_large_scale/results/v16_20260929/v15_torque_reward_eval_" + str(selection["evaluation_job"])
for hashes in candidate["data_hashes"].values():
    for filename, expected in hashes.items():
        assert hashlib.sha256((Path(selection["source_eval_dir"]) / filename).read_bytes()).hexdigest() == expected
print(json.dumps({"seed42_screening_all_gates_reverified": True, "selected_actor_sha256": candidate["config"]["actor_sha256"]}))
PYSELECT
mapfile -t model_paths < <(/opt/isaacsim-env/bin/python - "$audit_output/v15_torque_reward_eval_models.json" <<'PYMODEL'
import json, sys
paths = json.load(open(sys.argv[1]))
for label in ("adapt_torque1p25_g099_lpf25",):
    print(paths[label])
PYMODEL
)
[[ "${#model_paths[@]}" -eq 1 ]] || exit 2
sha256sum "$actor" "${model_paths[@]}" "$audit_output"/v15_torque_* "$audit_output"/reward_training_* \
  "$v15/config/evaluate_v13_torque.py" "$v15/config/v15_env_cfg.py" \
  /home/shanghun1404/v13_torque/config/v13_torque_actions.py \
  /home/shanghun1404/v13_torque/config/v13_torque_mdp.py \
  "$v14/config/v14_ros2_bridge.py" "$v14/assets/robot_v2_v14_leg_collision.usda" \
  /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt > "$audit_output/source_sha256.txt"
/opt/isaacsim-env/bin/python - "$audit_output/v15_torque_reward_eval_manifest_template.json" <<'PYREUSE' >> "$audit_output/source_sha256.txt"
import json, sys
reuse = json.load(open(sys.argv[1]))["baseline_reuse"]
for group in ("artifact_sha256", "immutable_source_sha256"):
    for path, digest in reuse[group].items():
        print(digest + "  " + path)
PYREUSE
cd "$v15/config"
baseline_actor=$actor
selected_actor=${model_paths[0]}
for candidate in baseline adapt_torque1p25_g099_lpf25; do
    actor=$selected_actor
    if [[ "$candidate" == baseline ]]; then
        actor=$baseline_actor
    fi
    guard=0.99
    cutoff=25
    export V15_TORQUE_CANDIDATE=$candidate
    export V15_TORQUE_CUTOFF_HZ=$cutoff
    for scenario in flat rough; do
        terrain_args=()
        heading_args=()
        speed=0.4
        warmup=2
        if [[ "$scenario" == rough ]]; then
            terrain_args=(--terrain-level 4)
            heading_args=(--heading-control)
            speed=0.5
            warmup=0
        fi
        output=$audit_output/${candidate}_${scenario}_seed${holdout_seed}
        test ! -e "$output.json"
        /opt/isaacsim-env/bin/python "$audit_output/v15_torque_reward_eval_capture.py" \
          --headless --enable_cameras --variant v13 --num-envs 32 --seed "$holdout_seed" \
          --warmup-seconds "$warmup" --seconds 12 --forward-speed "$speed" \
          --scenario "$scenario" "${terrain_args[@]}" "${heading_args[@]}" --protocol fixed_v2 \
          --joint-drive physx --effort-guard-fraction "$guard" \
          --enable-self-collisions --filter-leg-terrain --measure-leg-clearance \
          --actor-checkpoint "$actor" \
          --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
          --usd-path "$v14/assets/robot_v2_v14_leg_collision.usda" \
          --output "$output.json" --trajectory-output "${output}_trajectory.npz" \
          --trace-output "${output}_trace.npz" > "${output}.log" 2>&1
        test -s "$output.json"
        test -s "${output}_trace_audit.npz"
        sha256sum --quiet -c "$audit_output/source_sha256.txt"
        sha256sum "$output.json" "${output}_trace.npz" "${output}_trace_audit.npz" "${output}_trajectory.npz" >> "$audit_output/data_sha256.txt"
    done
done
