#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=v15-torque-sweep
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
set -euo pipefail
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
audit_config=$v15/config/experiments/v16_20260929
audit_output=$v15/results/v16_20260929/v15_torque_sweep_${SLURM_JOB_ID}
actor=$v15/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
mkdir -p "$audit_output"
cp "$audit_config/v15_torque_capture.py" "$audit_config/v15_torque_candidate_capture.py" "$audit_config/v15_torque_sweep.sbatch" "$audit_config/v15_torque_candidates.json" "$audit_output/"
sha256sum "$actor" "$audit_config"/v15_torque_*.py "$audit_config/v15_torque_candidates.json" "$audit_config/v15_torque_sweep.sbatch" > "$audit_output/source_sha256.txt"
cd "$v15/config"
for candidate in g095_lpf15 g090_lpf25 g090_lpf10; do
    case "$candidate" in
        g095_lpf15) guard=0.95; cutoff=15 ;;
        g090_lpf25) guard=0.90; cutoff=25 ;;
        g090_lpf10) guard=0.90; cutoff=10 ;;
    esac
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
        /opt/isaacsim-env/bin/python "$audit_config/v15_torque_candidate_capture.py" \
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
        sha256sum "$output.json" "${output}_trace.npz" "${output}_trace_audit.npz" "${output}_trajectory.npz" >> "$audit_output/data_sha256.txt"
    done
done
