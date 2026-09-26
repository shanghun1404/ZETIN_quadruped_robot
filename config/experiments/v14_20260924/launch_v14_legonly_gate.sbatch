#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=v14-legonly-gate
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config

actor=/home/projects/legacy/quadruped_training_smooth_v11/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
usd=/home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_radius_015_legonly.usda
results=/home/shanghun1404/v14_leg_collision/results

for scenario in flat stairs_up; do
  warmup=0
  speed=0.5
  terrain_args=(--terrain-level 5)
  if [[ "$scenario" == flat ]]; then
    warmup=2
    speed=0.4
    terrain_args=()
  fi
  /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
    --headless --enable_cameras --variant v13 --num-envs 32 --seed 42 \
    --warmup-seconds "$warmup" --seconds 12 --forward-speed "$speed" \
    --scenario "$scenario" "${terrain_args[@]}" \
    --effort-guard-fraction 0.99 --enable-self-collisions --measure-leg-clearance \
    --leg-capsule-radius-m 0.015 \
    --actor-checkpoint "$actor" --student-checkpoint "$student" --usd-path "$usd" \
    --output "$results/v14_legonly_${scenario}_seed42.json"
done
