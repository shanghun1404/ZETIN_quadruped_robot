#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13G
#SBATCH --time=00:45:00
#SBATCH --job-name=v6-cad-suite
#SBATCH --output=/home/shanghun1404/v15_cad_20261011/v6-suite-%j.out
# V6 round-body CAD (2026-10-25) under model_15699 (default) or a V6 fine-tuned actor (third argument).
# Evaluation source is the snapshot of round2 suite job 2343; only the robot USD and the actor differ from jobs 2342/2343.
set -euo pipefail
root=/home/shanghun1404/v15_cad_20261011
v14=/home/shanghun1404/v14_leg_collision
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
python=/opt/isaacsim-env/bin/python
cases=${1:?comma-separated case names required}
seed=${2:-42}
frozen=$root/suite_jobs/2343
asset=$root/assets/quadruped_v6_20261025
actor=${3:-$root/train_jobs/2341/results/model_15699.pt}
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
sha256sum -c --quiet "$frozen/source.sha256" "$frozen/checkpoints.sha256"
output=$root/v6_suite_jobs/$SLURM_JOB_ID
mkdir -p "$output/source/assets"
cp -r "$frozen/source/config" "$frozen/source/ros2" "$output/source/"
cp -r "$asset" "$output/source/assets/"
cp "$root/config/experiments/v6_cad_20261025/suite.sbatch" "$output/launcher.sbatch"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$output/source/config:$output/source/ros2/v151_ros2_control:$output/source/ros2/v14_ros2_control:/home/shanghun1404/v15_large_scale/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
$python -c 'import json,os,sys; json.dump({k:os.environ.get(k) for k in ("RMW_IMPLEMENTATION","ROS_DOMAIN_ID","ROS_DISTRO","SLURM_JOB_ID","SLURM_JOB_QOS","CUDA_VISIBLE_DEVICES")},open(sys.argv[1],"w"),indent=2)' "$output/runtime.json"
find "$output/source" -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$output/source.sha256"
sha256sum "$actor" "$student" > "$output/checkpoints.sha256"
exec "$python" "$output/source/config/evaluate_v15_cad_suite.py" --headless --enable_cameras \
 --usd-path "$output/source/assets/quadruped_v6_20261025/quadruped_robot.usda" \
 --actor "$actor" --student "$student" --model-path "$output/source/config/model/quadruped_cad.xml" \
 --output-dir "$output/results" --cases "$cases" --seed "$seed" > "$output/simulator.log" 2>&1
