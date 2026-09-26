#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=v14-actor-gate
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out

set -euo pipefail
actor=${1:?actor checkpoint is required}
label=${2:?result label is required}
test -s "$actor"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config

student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
usd=/home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda
results=/home/shanghun1404/v14_leg_collision/results/actor_gate
mkdir -p "$results"

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
    --effort-guard-fraction 0.99 --enable-self-collisions \
    --filter-leg-terrain --measure-leg-clearance \
    --actor-checkpoint "$actor" --student-checkpoint "$student" --usd-path "$usd" \
    --output "$results/${label}_${scenario}_seed42.json"
done
