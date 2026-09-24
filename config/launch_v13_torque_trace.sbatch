#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=v13-torque-trace
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

results=/home/shanghun1404/v13_torque/results/traces
mkdir -p "$results"
v11_actor=/home/projects/legacy/quadruped_training_smooth_v11/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt
v12_student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
robot_usd=/home/projects/legacy/robot_v2/usd/v2_plain.usd
v13_actor=$(cat /home/shanghun1404/v13_torque/results/v13_pilot_checkpoint.txt)
test -s "$v13_actor"

for variant in v12 v13; do
  actor=$v11_actor
  if [[ "$variant" == v13 ]]; then actor=$v13_actor; fi
  /opt/isaacsim-env/bin/python evaluate_v13_torque.py \
    --headless --enable_cameras --variant "$variant" --num-envs 32 --seed 42 \
    --warmup-seconds 2 --seconds 12 --forward-speed 0.4 \
    --actor-checkpoint "$actor" --student-checkpoint "$v12_student" \
    --usd-path "$robot_usd" \
    --output "$results/"$variant"_seed42.json" \
    --trace-output "$results/"$variant"_seed42.npz"
  test -s "$results/"$variant"_seed42.npz"
done
