#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=v15-cad-finetune
#SBATCH --output=/home/shanghun1404/v15_cad_20261011/train-%j.out
set -euo pipefail
root=/home/shanghun1404/v15_cad_20261011
job=$root/train_jobs/$SLURM_JOB_ID
num_envs=${1:-128}
iterations=${2:-2000}
terrain=${3:-mixed}
actor=${4:-/home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt}
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
mkdir -p "$job/source/config" "$job/source/assets" "$job/source/ros2"
cp "$root/config/"*.py "$job/source/config/"
cp -r "$root/config/experiments/v15_cad_20261011/model" "$job/source/config/"
cp -r "$root/assets/quadruped_20261011" "$job/source/assets/"
cp -r "$root/ros2/v151_ros2_control" "$root/ros2/v14_ros2_control" "$job/source/ros2/"
cp "$root/config/experiments/v15_cad_20261011/finetune.sbatch" "$job/launcher.sbatch"
find "$job/source" -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$job/source.sha256"
sha256sum "$actor" "$student" > "$job/checkpoints.sha256"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$job/source/config:$job/source/ros2/v151_ros2_control:$job/source/ros2/v14_ros2_control:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
/opt/isaacsim-env/bin/python -c 'import json,os,sys;json.dump({k:os.environ.get(k) for k in ("SLURM_JOB_ID","SLURM_JOB_QOS","CUDA_VISIBLE_DEVICES","ROS_DOMAIN_ID")},open(sys.argv[1],"w"),indent=2)' "$job/runtime.json"
exec /opt/isaacsim-env/bin/python "$job/source/config/train_v15_cad.py" --headless --enable_cameras \
    --num_envs "$num_envs" --iterations "$iterations" --terrain "$terrain" --seed 73 \
    --actor "$actor" --student "$student" --usd "$job/source/assets/quadruped_20261011/quadruped_robot.usda" \
    --output "$job/results" > "$job/training.log" 2>&1
