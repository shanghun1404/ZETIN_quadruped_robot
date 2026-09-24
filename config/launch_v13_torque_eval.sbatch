#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=v13-torque-eval
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

checkpoint=$(cat /home/shanghun1404/v13_torque/results/v13_pilot_checkpoint.txt)
test -s "$checkpoint"

mkdir -p /home/shanghun1404/v13_torque/results
for seed in 42 43 44; do
  /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
    --headless --enable_cameras --variant v12 --num-envs 32 --seed "$seed" \
    --actor-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt \
    --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
    --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
    --output "/home/shanghun1404/v13_torque/results/v12_flat_seed${seed}.json"
  /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
    --headless --enable_cameras --variant v13 --num-envs 32 --seed "$seed" \
    --actor-checkpoint "$checkpoint" \
    --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
    --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
    --output "/home/shanghun1404/v13_torque/results/v13_flat_seed${seed}.json"
done
