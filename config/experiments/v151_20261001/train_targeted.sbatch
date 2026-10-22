#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=v151-targeted
#SBATCH --output=/home/shanghun1404/v151_ros2/slurm-targeted-%j.out
set -euo pipefail
root=/home/shanghun1404/v151_ros2
source_root=$root/targeted_20261001/config
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=$source_root:$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
cd "$source_root"
sha256sum --check train_targeted_sources.sha256
/opt/isaacsim-env/bin/python -m pytest -q tests/test_v151_targeted.py tests/test_v151_training_math.py
/opt/isaacsim-env/bin/python train_v151_targeted.py \
  --headless --enable_cameras --resume --num_envs 512 --max_iterations 100 \
  --seed 47 --max-delay-steps 3 --output-root "$root/targeted_training_jobs/$SLURM_JOB_ID" \
  --actor-checkpoint /home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path "$v14/assets/robot_v2_v14_leg_collision.usda"
