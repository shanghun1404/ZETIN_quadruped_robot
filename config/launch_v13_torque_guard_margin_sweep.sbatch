#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=v13-guard-margin
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

checkpoint=$(cat /home/shanghun1404/v13_torque/results/v13_pilot_checkpoint.txt)
test -s "$checkpoint"
results=/home/shanghun1404/v13_torque/results/guard_margin_sweep
mkdir -p "$results"

for fraction in 0.99 1.0; do
  /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
    --headless --enable_cameras --variant v13 --num-envs 32 --seed 42 \
    --warmup-seconds 0 --seconds 12 --forward-speed 0.5 \
    --scenario rough --terrain-level 4 --effort-guard-fraction "$fraction" \
    --actor-checkpoint "$checkpoint" \
    --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
    --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
    --output "$results/v13_guard_${fraction}_rough_seed42.json"
done
