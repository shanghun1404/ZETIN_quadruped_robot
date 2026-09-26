#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=v14-leg-physics
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config

source_usd=/home/projects/legacy/robot_v2/usd/v2_plain.usd
derived_usd=/home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda
results=/home/shanghun1404/v14_leg_collision/results

/opt/isaacsim-env/bin/python probe_v14_leg_physics.py \
  --headless --mode cross --usd-path "$source_usd" \
  --output "$results/original_cross_reset.json"
/opt/isaacsim-env/bin/python probe_v14_leg_physics.py \
  --headless --mode cross --usd-path "$derived_usd" --self-collisions \
  --output "$results/v14_cross_reset.json"
