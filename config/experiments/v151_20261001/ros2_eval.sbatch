#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=v151-ros2
#SBATCH --output=/home/shanghun1404/v151_ros2/slurm-ros2-%j.out
set -euo pipefail
root=/home/shanghun1404/v151_ros2
v14=/home/shanghun1404/v14_leg_collision
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
python=/opt/isaacsim-env/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$root/config:$root/ros2/v151_ros2_control:$v14/ros2/v14_ros2_control:/home/shanghun1404/v15_large_scale/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp} ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
actor=${1:-/home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt}
mode=${2:-demo}
scenario=${3:-flat}
seed=${4:-42}
seconds=${5:-28}
forward_shift=${6:-0.03}
crouch_forward_shift=${7:-0.15}
crouch_shift_after_walk_only=${8:-true}
output=$root/evaluation_jobs/$SLURM_JOB_ID
mkdir -p "$output/source/config" "$output/source/ros2"
$python -c 'import json,os,sys; json.dump({k:os.environ.get(k) for k in ("RMW_IMPLEMENTATION","ROS_DOMAIN_ID","ROS_DISTRO","FASTDDS_BUILTIN_TRANSPORTS","SLURM_JOB_ID","SLURM_JOB_QOS")},open(sys.argv[1],"w"),indent=2)' "$output/runtime.json"
cp "$root/config/"*.py "$output/source/config/"
cp -r "$root/ros2/v151_ros2_control" "$output/source/ros2/"
cp "$root/config/experiments/v151_20261001/ros2_eval.sbatch" "$output/launcher.sbatch"
export PYTHONPATH=$output/source/config:$output/source/ros2/v151_ros2_control:$v14/ros2/v14_ros2_control:/home/shanghun1404/v15_large_scale/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
find "$output/source" -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$output/source.sha256"
sha256sum "$actor" "$root/model/quadruped.xml" "$root/model/crouch.json" > "$output/inputs.sha256"
model=$root/model/quadruped.xml
export V151_MODEL=$model
$python -m pytest -q "$output/source/ros2/v151_ros2_control/test/test_motion.py" "$output/source/ros2/v151_ros2_control/test/test_standing.py" > "$output/tests.log" 2>&1
initial=SIT
demo=true
pose=(--initial-pose "$root/model/crouch.json")
if [[ "$mode" == walk ]]; then initial=WALK; demo=false; pose=(); fi
controller_pid=
simulator_pid=
trap 'kill ${controller_pid:-} ${simulator_pid:-} 2>/dev/null || true' EXIT
$python "$output/source/config/evaluate_v151_ros2.py" --headless --enable_cameras \
 --usd-path "$v14/assets/robot_v2_v14_leg_collision.usda" --output "$output/result.json" \
 --seconds "$seconds" --seed "$seed" --scenario "$scenario" "${pose[@]}" \
 --video-dir "$output/video" &
simulator_pid=$!
while [[ ! -f "$output/result.controller_start" ]]; do
    if ! kill -0 "$simulator_pid" 2>/dev/null; then wait "$simulator_pid"; exit 1; fi
    sleep 0.2
done
$python -m v151_ros2_control.controller_node --ros-args -r __ns:=/v151 \
 -p actor_checkpoint:="$actor" -p student_checkpoint:=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
 -p model_path:="$model" -p stand_forward_shift_m:="$forward_shift" -p crouch_forward_shift_m:="$crouch_forward_shift" -p crouch_shift_after_walk_only:="$crouch_shift_after_walk_only" -p initial_mode:="$initial" -p demo:="$demo" > "$output/controller.log" 2>&1 &
controller_pid=$!
wait "$simulator_pid"
