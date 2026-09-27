#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=v14-ros2-eval
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out
# One-robot fixed_v2 evaluation of the V14 5700 actor + V12 depth student.
#   control=ros2:     ROS2 policy + LPF/guard nodes, Isaac Sim PhysX joint-drive PD (lockstep)
#   control=isaaclab: the same single-robot run with the original in-process IsaacLab control
set -euo pipefail
control=${1:?control (ros2|isaaclab) required}
label=${2:?label required}
scenario=${3:-flat}
seed=${4:-42}
root=/home/shanghun1404/v14_leg_collision
actor=${5:-$root/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt}
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
usd=$root/assets/robot_v2_v14_leg_collision.usda
python=/opt/isaacsim-env/bin/python
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$root/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$root/ros2/v14_ros2_control:$ros/rclpy
# Isaac Sim's bundled ROS2 Jazzy libraries; no system ROS2 is installed on ws1.
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$(( ${SLURM_JOB_ID:-0} % 90 + 10 ))
cd $root/config

output=$root/results/ros2/${label}_${control}_${scenario}_seed${seed}
test ! -e "$output.json"
mkdir -p "$(dirname "$output")"
terrain_args=(--terrain-level 5)
speed=0.5
warmup=0
if [[ "$scenario" == flat ]]; then
    terrain_args=()
    speed=0.4
    warmup=2
elif [[ "$scenario" == rough ]]; then
    terrain_args=(--terrain-level 4)
fi

control_args=(--control "$control")
if [[ "$control" == ros2 ]]; then
    pids=()
    trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
    $python -m v14_ros2_control.joint_command_node --ros-args -r __ns:=/v14 \
        -p guard_fraction:=0.99 -p cutoff_hz:=25.0 > "$output.joint_command.log" 2>&1 &
    pids+=($!)
    $python -m v14_ros2_control.policy_node --ros-args -r __ns:=/v14 \
        -p actor_checkpoint:="$actor" -p student_checkpoint:="$student" \
        -p device:=cpu -p "command:=[0.0, $speed, 0.0]" > "$output.policy.log" 2>&1 &
    pids+=($!)
elif [[ "$control" == isaaclab ]]; then
    control_args+=(--effort-guard-fraction 0.99)
else
    echo "unsupported control: $control" >&2
    exit 2
fi

$python evaluate_v13_torque.py \
  --headless --enable_cameras --variant v13 --num-envs 1 --seed "$seed" \
  --warmup-seconds "$warmup" --seconds 12 --forward-speed "$speed" \
  --scenario "$scenario" "${terrain_args[@]}" --protocol fixed_v2 "${control_args[@]}" \
  --enable-self-collisions --filter-leg-terrain --measure-leg-clearance \
  --actor-checkpoint "$actor" --student-checkpoint "$student" --usd-path "$usd" \
  --output "$output.json" --trajectory-output "$output.npz"
test -s "$output.json"
test -s "$output.npz"
