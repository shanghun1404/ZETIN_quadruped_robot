#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=v13-torque-pilot
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

pilot_root=/home/shanghun1404/v13_torque/pilot_jobs/${SLURM_JOB_ID}
/opt/isaacsim-env/bin/python train_v13_torque.py \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --max_iterations 1500 \
  --seed 42 \
  --output-root "$pilot_root" \
  --actor-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd

run_dir=$(find "$pilot_root/guide_v13_v12_depth_torque" -mindepth 1 -maxdepth 1 -type d -print -quit)
test -n "$run_dir"
checkpoint=$(find "$run_dir" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)
test -n "$checkpoint"
test -s "$run_dir/$checkpoint"
mkdir -p /home/shanghun1404/v13_torque/results
printf '%s\n' "$run_dir/$checkpoint" > /home/shanghun1404/v13_torque/results/v13_pilot_checkpoint.txt
echo "[INFO] Verified V13 pilot checkpoint: $run_dir/$checkpoint"
