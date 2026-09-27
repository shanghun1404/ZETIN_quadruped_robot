#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=v15-smoke
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# Two-iteration, 64-robot runs of both V15 trainers; gate long jobs on this after code changes.
set -euo pipefail
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
cd $v15/config
root=$v15/smoke_jobs/${SLURM_JOB_ID}
/opt/isaacsim-env/bin/python train_v15_recovery.py --headless --num_envs 64 --max_iterations 2 \
  --output-root "$root" --usd-path $v15/assets/robot_v2_v15_recovery.usda
/opt/isaacsim-env/bin/python train_v15_locomotion.py --headless --enable_cameras --num_envs 64 --max_iterations 2 \
  --output-root "$root" \
  --actor-checkpoint $v14/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path $v14/assets/robot_v2_v14_leg_collision.usda
test -n "$(find "$root/guide_v15_recovery" -name 'model_*.pt' -print -quit)"
test -n "$(find "$root/guide_v15_physx_pd" -name 'model_*.pt' -print -quit)"
