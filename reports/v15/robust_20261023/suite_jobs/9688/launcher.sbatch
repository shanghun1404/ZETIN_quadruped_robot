#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13G
#SBATCH --time=02:00:00
#SBATCH --job-name=v15-robust-suite
#SBATCH --output=/home/shanghun1404/v15_robust_20261023/suite-%j.out
set -euo pipefail
root=/home/shanghun1404/v15_robust_20261023
v14=/home/shanghun1404/v14_leg_collision
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
python=/opt/isaacsim-env/bin/python
cases=${1:?comma-separated case names required}
actor=${2:-/home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt}
seed=${3:-42}
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
output=$root/suite_jobs/$SLURM_JOB_ID
mkdir -p "$output/source/config" "$output/source/ros2" "$output/source/assets"
cp "$root/config/"*.py "$output/source/config/"
cp -r "$root/config/experiments/v15_cad_20261011/model" "$output/source/config/"
cp -r "$root/ros2/v151_ros2_control" "$root/ros2/v14_ros2_control" "$output/source/ros2/"
cp -r "$root/assets/quadruped_20261011" "$output/source/assets/"
cp "$root/config/experiments/v15_robust_20261023/suite.sbatch" "$output/launcher.sbatch"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$output/source/config:$output/source/ros2/v151_ros2_control:$output/source/ros2/v14_ros2_control:/home/shanghun1404/v15_large_scale/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
$python -c 'import json,os,sys; json.dump({k:os.environ.get(k) for k in ("RMW_IMPLEMENTATION","ROS_DOMAIN_ID","ROS_DISTRO","SLURM_JOB_ID","SLURM_JOB_QOS","CUDA_VISIBLE_DEVICES")},open(sys.argv[1],"w"),indent=2)' "$output/runtime.json"
find "$output/source" -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$output/source.sha256"
sha256sum "$actor" "$student" > "$output/checkpoints.sha256"
exec "$python" "$output/source/config/evaluate_v15_cad_suite.py" --headless --enable_cameras \
 --usd-path "$output/source/assets/quadruped_20261011/quadruped_robot.usda" \
 --actor "$actor" --student "$student" --model-path "$output/source/config/model/quadruped_cad.xml" \
 --output-dir "$output/results" --cases "$cases" --seed "$seed" > "$output/simulator.log" 2>&1
