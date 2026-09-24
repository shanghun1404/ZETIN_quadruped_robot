#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=v13-v11-guard-validation
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

actor=/home/projects/legacy/quadruped_training_smooth_v11/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt
test -s "$actor"
results=/home/shanghun1404/v13_torque/results/v11_actor_guard_099
mkdir -p "$results"

for seed in 42 43 44; do
  /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
    --headless --enable_cameras --variant v13 --num-envs 32 --seed "$seed" \
    --warmup-seconds 2 --seconds 12 --forward-speed 0.4 \
    --effort-guard-fraction 0.99 \
    --actor-checkpoint "$actor" \
    --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
    --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
    --output "$results/v11_actor_guard_0.99_flat_seed${seed}.json"
done

for scenario in rough stairs_up stairs_down; do
  seeds=(43 44)
  if [[ "$scenario" == stairs_down ]]; then seeds=(42 43 44); fi
  for seed in "${seeds[@]}"; do
    /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
      --headless --enable_cameras --variant v13 --num-envs 32 --seed "$seed" \
      --warmup-seconds 0 --seconds 12 --forward-speed 0.5 \
      --scenario "$scenario" --effort-guard-fraction 0.99 \
      --actor-checkpoint "$actor" \
      --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
      --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
      --output "$results/v11_actor_guard_0.99_${scenario}_seed${seed}.json"
  done
done
