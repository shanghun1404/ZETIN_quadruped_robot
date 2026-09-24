#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=v13-torque-smoke
#SBATCH --output=/home/shanghun1404/v13_torque/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v13_torque/config

smoke_root=/home/shanghun1404/v13_torque/smoke_jobs/${SLURM_JOB_ID}
/opt/isaacsim-env/bin/python train_v13_torque.py \
  --headless \
  --enable_cameras \
  --num_envs 16 \
  --max_iterations 3 \
  --seed 42 \
  --output-root "$smoke_root" \
  --actor-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/projects/legacy/robot_v2/usd/v2_plain.usd

checkpoint=$(find "$smoke_root" -type f -name 'model_*.pt' -print -quit)
test -n "$checkpoint"
test -s "$checkpoint"
echo "[INFO] Verified V13 smoke checkpoint: $checkpoint"
