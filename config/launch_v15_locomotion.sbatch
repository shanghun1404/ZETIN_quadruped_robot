#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=v15-locomotion
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# V15-L: fine-tune the V14 5700 actor on PhysX joint-drive PD (see docs/v15_design.md).
#   sbatch launch_v15_locomotion.sbatch <num_envs> <iterations> [V15 checkpoint to resume]
set -euo pipefail
num_envs=${1:?num_envs required}
iterations=${2:?iterations required}
resume=${3:-}
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
cd $v15/config
run_root=$v15/locomotion_jobs/${SLURM_JOB_ID}
actor=$v14/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt
resume_args=()
if [[ -n "$resume" ]]; then
    test -s "$resume"
    actor=$resume
    resume_args=(--resume)
fi
/opt/isaacsim-env/bin/python train_v15_locomotion.py \
  --headless --enable_cameras --num_envs "$num_envs" --max_iterations "$iterations" \
  --output-root "$run_root" "${resume_args[@]}" \
  --actor-checkpoint "$actor" \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path $v14/assets/robot_v2_v14_leg_collision.usda
run_dir=$(find "$run_root/guide_v15_physx_pd" -mindepth 1 -maxdepth 1 -type d -print -quit)
checkpoint=$(find "$run_dir" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)
test -s "$run_dir/$checkpoint"
printf '%s\n' "$run_dir/$checkpoint" > "$run_root/checkpoint.txt"
