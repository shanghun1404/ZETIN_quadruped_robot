#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=v14-stair-smoke
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config

actor=$(find /home/shanghun1404/v14_leg_collision/pilot_jobs/110 -name model_5700.pt -print -quit)
test -s "$actor"
/opt/isaacsim-env/bin/python train_v14_leg_collision.py \
  --headless --enable_cameras --num_envs 16 --max_iterations 1 --seed 42 \
  --initial-terrain-level 4 --stair-focus \
  --output-root /home/shanghun1404/v14_leg_collision/stair_smoke/${SLURM_JOB_ID} \
  --actor-checkpoint "$actor" \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda
