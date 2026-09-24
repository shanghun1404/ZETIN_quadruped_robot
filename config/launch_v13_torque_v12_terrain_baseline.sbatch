#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=v13-v12-terrain-baseline
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

actor=/home/projects/legacy/quadruped_training_smooth_v11/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt
test -s "$actor"
results=/home/shanghun1404/v13_torque/results/v12_terrain_baseline
mkdir -p "$results"

for scenario in rough stairs_up stairs_down; do
  seeds=(43 44)
  if [[ "$scenario" == stairs_down ]]; then seeds=(42 43 44); fi
  for seed in "${seeds[@]}"; do
    /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
      --headless --enable_cameras --variant v12 --num-envs 32 --seed "$seed" \
      --warmup-seconds 0 --seconds 12 --forward-speed 0.5 \
      --scenario "$scenario" \
      --actor-checkpoint "$actor" \
      --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
      --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
      --output "$results/v12_${scenario}_seed${seed}.json"
  done
done
