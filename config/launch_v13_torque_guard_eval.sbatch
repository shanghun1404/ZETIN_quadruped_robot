#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=v13-guard-eval
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

checkpoint=$(cat /home/shanghun1404/v13_torque/results/v13_pilot_checkpoint.txt)
test -s "$checkpoint"
results=/home/shanghun1404/v13_torque/results/guard_095
mkdir -p "$results"

for seed in 42 43 44; do
  /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
    --headless --enable_cameras --variant v13 --num-envs 32 --seed "$seed" \
    --warmup-seconds 2 --seconds 12 --forward-speed 0.4 \
    --effort-guard-fraction 0.95 \
    --actor-checkpoint "$checkpoint" \
    --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
    --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
    --output "$results/v13_guard_flat_seed${seed}.json"
done
