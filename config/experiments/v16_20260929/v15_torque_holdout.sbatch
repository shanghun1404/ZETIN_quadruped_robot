#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=v15-torque-holdout
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# One seed per allocation: fresh flat/rough baseline and unchanged selected candidate.
set -euo pipefail
selected=${1:?selected candidate required}
holdout_seed=${2:?one holdout seed required (43 or 44)}
[[ "$holdout_seed" == 43 || "$holdout_seed" == 44 ]] || exit 2
case "$selected" in
    g095_lpf15) selected_guard=0.95; selected_cutoff=15 ;;
    g090_lpf25) selected_guard=0.90; selected_cutoff=25 ;;
    g090_lpf10) selected_guard=0.90; selected_cutoff=10 ;;
    g095_lpf20) selected_guard=0.95; selected_cutoff=20 ;;
    g097_lpf20) selected_guard=0.97; selected_cutoff=20 ;;
    *) exit 2 ;;
esac
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
audit_config=$v15/config/experiments/v16_20260929
audit_output=$v15/results/v16_20260929/v15_torque_holdout_${SLURM_JOB_ID}
actor=$v15/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
mkdir -p "$audit_output"
cp "$audit_config/v15_torque_capture.py" "$audit_config/v15_torque_candidate_capture.py" "$audit_config/v15_torque_refinement_capture.py" "$audit_config/v15_torque_holdout.sbatch" "$audit_config/v15_torque_candidates.json" "$audit_config/v15_torque_refinement_candidates.json" "$audit_config/v15_torque_holdout_selection.json" "$audit_config/v15_torque_holdout_protocol.json" "$audit_output/"
sha256sum "$actor" "$audit_output"/v15_torque_* > "$audit_output/source_sha256.txt"
/opt/isaacsim-env/bin/python - "$audit_output/v15_torque_holdout_selection.json" "$selected" "$selected_guard" "$selected_cutoff" <<'PY'
import json, sys
selection = json.load(open(sys.argv[1]))
assert selection["selected_candidate"] == sys.argv[2]
assert selection["config"]["guard_fraction"] == float(sys.argv[3])
assert selection["config"]["cutoff_hz"] == float(sys.argv[4])
assert selection["seed42_all_gates_pass"] is True
PY
cd "$v15/config"
for seed in "$holdout_seed"; do
    for condition in baseline_flat baseline_rough selected_flat selected_rough; do
        scenario=rough
        terrain_args=(--terrain-level 4)
        heading_args=(--heading-control)
        speed=0.5
        warmup=0
        label=$selected
        guard=$selected_guard
        cutoff=$selected_cutoff
        capture=$audit_output/v15_torque_candidate_capture.py
        if [[ "$selected_cutoff" -eq 20 ]]; then
            capture=$audit_output/v15_torque_refinement_capture.py
        fi
        if [[ "$condition" == baseline_* ]]; then
            label=baseline
            guard=0.99
            cutoff=25
            capture=$audit_output/v15_torque_capture.py
        fi
        if [[ "$condition" == *_flat ]]; then
            scenario=flat
            terrain_args=()
            heading_args=()
            speed=0.4
            warmup=2
        fi
        export V15_TORQUE_CANDIDATE=$label
        export V15_TORQUE_CUTOFF_HZ=$cutoff
        output=$audit_output/${label}_${scenario}_seed${seed}
        test ! -e "$output.json"
        /opt/isaacsim-env/bin/python "$capture" \
          --headless --enable_cameras --variant v13 --num-envs 32 --seed "$seed" \
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
        sha256sum "$output.json" "${output}_trace.npz" "${output}_trace_audit.npz" "${output}_trajectory.npz" >> "$audit_output/data_sha256.txt"
    done
done
