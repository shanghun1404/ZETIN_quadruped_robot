#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=v14-direction-train
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out
set -euo pipefail
iterations=${1:-300}
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config
pilot_root=/home/shanghun1404/v14_leg_collision/direction_jobs/${SLURM_JOB_ID}
/opt/isaacsim-env/bin/python train_v14_leg_collision.py \
  --headless --enable_cameras --num_envs 32 --max_iterations "$iterations" --seed 45 \
  --initial-terrain-level 0 --direction-focus --output-root "$pilot_root" \
  --actor-checkpoint /home/shanghun1404/v14_leg_collision/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda
run_dir=$(find "$pilot_root/guide_v14_direction_adaptation" -mindepth 1 -maxdepth 1 -type d -print -quit)
test -n "$run_dir"
checkpoint=$(find "$run_dir" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)
test -n "$checkpoint"
test -s "$run_dir/$checkpoint"
printf '%s\n' "$run_dir/$checkpoint" > "$pilot_root/checkpoint.txt"
