#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=v13-guard-terrain
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

checkpoint=$(cat /home/shanghun1404/v13_torque/results/v13_pilot_checkpoint.txt)
test -s "$checkpoint"
results=/home/shanghun1404/v13_torque/results/guard_095_terrain
mkdir -p "$results"

for scenario in rough stairs_up stairs_down; do
  for variant in baseline guard; do
    guard_args=()
    if [[ "$variant" == guard ]]; then guard_args=(--effort-guard-fraction 0.95); fi
    /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
      --headless --enable_cameras --variant v13 --num-envs 32 --seed 42 \
      --warmup-seconds 0 --seconds 12 --forward-speed 0.5 \
      --scenario "$scenario" "${guard_args[@]}" \
      --actor-checkpoint "$checkpoint" \
      --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
      --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
      --output "$results/v13_${variant}_${scenario}_seed42.json"
  done
done
