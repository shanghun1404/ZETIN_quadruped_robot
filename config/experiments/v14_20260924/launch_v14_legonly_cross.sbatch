#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=v14-legonly-cross
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config

/opt/isaacsim-env/bin/python probe_v14_leg_physics.py \
  --headless --mode cross --self-collisions --capsule-radius-m 0.015 \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_radius_015_legonly.usda \
  --output /home/shanghun1404/v14_leg_collision/results/v14_legonly_cross.json
