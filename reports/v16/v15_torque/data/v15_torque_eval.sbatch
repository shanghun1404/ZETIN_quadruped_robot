#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=v15-torque-audit
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# One allocation, three sequential evaluations. Original launch_v15_eval conditions.
set -euo pipefail
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
audit_config=$v15/config/experiments/v16_20260929
audit_output=$v15/results/v16_20260929/v15_torque_${SLURM_JOB_ID}
actor=$v15/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
mkdir -p "$audit_output"
cp "$audit_config/v15_torque_capture.py" "$audit_config/v15_torque_eval.sbatch" "$audit_output/"
sha256sum "$actor" "$v15/config/evaluate_v13_torque.py" "$v15/config/v15_env_cfg.py" \
  /opt/isaaclab/source/isaaclab/isaaclab/actuators/actuator_pd.py \
  /opt/isaaclab/source/isaaclab/isaaclab/assets/articulation/articulation.py > "$audit_output/source_sha256.txt"
for scenario in flat stairs_up rough; do
    terrain_args=(--terrain-level 5)
    heading_args=(--heading-control)
    speed=0.5
    warmup=0
    if [[ "$scenario" == flat ]]; then
        terrain_args=()
        heading_args=()
        speed=0.4
        warmup=2
    elif [[ "$scenario" == rough ]]; then
        terrain_args=(--terrain-level 4)
    fi
    output=$audit_output/main339_${scenario}_seed42
    test ! -e "$output.json"
    /opt/isaacsim-env/bin/python "$audit_config/v15_torque_capture.py" \
      --headless --enable_cameras --variant v13 --num-envs 32 --seed 42 \
      --warmup-seconds "$warmup" --seconds 12 --forward-speed "$speed" \
      --scenario "$scenario" "${terrain_args[@]}" "${heading_args[@]}" --protocol fixed_v2 \
      --joint-drive physx --effort-guard-fraction 0.99 \
      --enable-self-collisions --filter-leg-terrain --measure-leg-clearance \
      --actor-checkpoint "$actor" \
      --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
      --usd-path "$v14/assets/robot_v2_v14_leg_collision.usda" \
      --output "$output.json" --trajectory-output "${output}_trajectory.npz" \
      --trace-output "${output}_trace.npz" > "${output}.log" 2>&1
    test -s "$output.json"
    test -s "${output}_trace_audit.npz"
done
sha256sum "$audit_output"/*.json "$audit_output"/*.npz > "$audit_output/data_sha256.txt"
