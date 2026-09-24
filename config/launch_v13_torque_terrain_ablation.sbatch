#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=v13-terrain-ablation
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

v11_actor=/home/projects/legacy/quadruped_training_smooth_v11/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt
v13_actor=$(cat /home/shanghun1404/v13_torque/results/v13_pilot_checkpoint.txt)
test -s "$v11_actor"
test -s "$v13_actor"
results=/home/shanghun1404/v13_torque/results/terrain_ablation
mkdir -p "$results"

for scenario in rough stairs_up; do
  for actor_name in v11 v13; do
    actor=$v11_actor
    if [[ "$actor_name" == v13 ]]; then actor=$v13_actor; fi
    /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
      --headless --enable_cameras --variant v12 --num-envs 32 --seed 42 \
      --warmup-seconds 0 --seconds 12 --forward-speed 0.5 \
      --scenario "$scenario" \
      --actor-checkpoint "$actor" \
      --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
      --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd \
      --output "$results/${actor_name}_actor_v12_control_${scenario}_seed42.json"
  done
done
