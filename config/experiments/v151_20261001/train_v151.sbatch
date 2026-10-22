#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=v151-rated-load
#SBATCH --output=/home/shanghun1404/v151_ros2/slurm-training-%j.out
set -euo pipefail
root=/home/shanghun1404/v151_ros2
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH=$root/config:$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
cd "$root/config"
sha256sum --check "$root/config/experiments/v151_20261001/train_sources.sha256"
/opt/isaacsim-env/bin/python train_v151_locomotion.py \
  --headless --enable_cameras --resume --num_envs 512 --max_iterations 500 \
  --seed 47 --max-delay-steps 3 --output-root "$root/training_jobs/$SLURM_JOB_ID" \
  --actor-checkpoint /home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path "$v14/assets/robot_v2_v14_leg_collision.usda"
