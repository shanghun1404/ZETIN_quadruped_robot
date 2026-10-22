#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=v15-torque-reward-eval
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
set -euo pipefail
training_job=${1:?completed torque reward training job ID required}
[[ "$training_job" =~ ^[0-9]+$ ]] || exit 2
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
audit_config=$v15/config/experiments/v16_20260929
audit_output=$v15/results/v16_20260929/v15_torque_reward_eval_${SLURM_JOB_ID}
actor=$v15/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
mkdir -p "$audit_output"
cp "$audit_config/v15_torque_capture.py" "$audit_config/v15_torque_reward_eval_capture.py" "$audit_config/v15_torque_reward_eval_manifest.py" "$audit_config/v15_torque_reward_eval_manifest_template.json" "$audit_config/v15_torque_reward_eval.sbatch" "$audit_output/"
/opt/isaacsim-env/bin/python "$audit_output/v15_torque_reward_eval_manifest.py" \
  --training-dir "$v15/results/v16_20260929/v15_torque_reward_finetune_${training_job}" \
  --template "$audit_output/v15_torque_reward_eval_manifest_template.json" --output-dir "$audit_output"
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
candidate=adapt_torque1p25_g099_lpf25
actor=${model_paths[0]}
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
        output=$audit_output/${candidate}_${scenario}_seed42
        test ! -e "$output.json"
        /opt/isaacsim-env/bin/python "$audit_output/v15_torque_reward_eval_capture.py" \
          --headless --enable_cameras --variant v13 --num-envs 32 --seed 42 \
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
