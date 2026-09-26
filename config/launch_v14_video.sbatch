#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=v14-video
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out

set -euo pipefail
label=${1:?new recording label required}
success_env=${2:?successful environment ID required}
failure_env=${3:?failed environment ID required}
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config
result=/home/shanghun1404/v14_leg_collision/results/videos/$label
test ! -e "$result"
mkdir -p "$result"
/opt/isaacsim-env/bin/python evaluate_v13_torque.py \
  --headless --enable_cameras --variant v13 --num-envs 32 --seed 42 \
  --warmup-seconds 0 --seconds 12 --forward-speed 0.5 \
  --scenario stairs_up --terrain-level 5 \
  --effort-guard-fraction 0.99 --enable-self-collisions \
  --filter-leg-terrain --measure-leg-clearance \
  --actor-checkpoint /home/shanghun1404/v14_leg_collision/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
  --output "$result/evaluation.json" \
  --video-dir "$result/recordings" --video-env-ids "$success_env" "$failure_env"
test -s "$result/evaluation.json"
