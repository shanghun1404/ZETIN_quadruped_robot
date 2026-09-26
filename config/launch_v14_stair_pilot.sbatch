#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=v14-stair-pilot
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config

actor=$(find /home/shanghun1404/v14_leg_collision/pilot_jobs/110 -name model_5700.pt -print -quit)
test -s "$actor"
pilot_root=/home/shanghun1404/v14_leg_collision/stair_jobs/${SLURM_JOB_ID}
/opt/isaacsim-env/bin/python train_v14_leg_collision.py \
  --headless --enable_cameras --num_envs 64 --max_iterations 800 --seed 43 \
  --initial-terrain-level 4 --stair-focus \
  --output-root "$pilot_root" \
  --actor-checkpoint "$actor" \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda

run_dir=$(find "$pilot_root/guide_v14_stair_adaptation" -mindepth 1 -maxdepth 1 -type d -print -quit)
test -n "$run_dir"
checkpoint=$(find "$run_dir" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)
test -n "$checkpoint"
test -s "$run_dir/$checkpoint"
printf '%s\n' "$run_dir/$checkpoint" > /home/shanghun1404/v14_leg_collision/results/v14_stair_checkpoint.txt
echo "[INFO] Verified V14 stair checkpoint: $run_dir/$checkpoint"
