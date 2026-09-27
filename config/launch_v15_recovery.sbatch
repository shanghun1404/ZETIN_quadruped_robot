#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=v15-recovery
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# V15-R: train the blind fall-recovery policy from scratch (see docs/v15_design.md).
#   sbatch launch_v15_recovery.sbatch <num_envs> <iterations> [V15-R checkpoint to resume]
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
run_root=$v15/recovery_jobs/${SLURM_JOB_ID}
resume_args=()
if [[ -n "$resume" ]]; then
    test -s "$resume"
    resume_args=(--resume-checkpoint "$resume")
fi
/opt/isaacsim-env/bin/python train_v15_recovery.py --headless --num_envs "$num_envs" \
  --max_iterations "$iterations" --output-root "$run_root" "${resume_args[@]}" \
  --usd-path $v15/assets/robot_v2_v15_recovery.usda
run_dir=$(find "$run_root/guide_v15_recovery" -mindepth 1 -maxdepth 1 -type d -print -quit)
checkpoint=$(find "$run_dir" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)
test -s "$run_dir/$checkpoint"
printf '%s\n' "$run_dir/$checkpoint" > "$run_root/checkpoint.txt"
