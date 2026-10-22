#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13G
#SBATCH --time=02:00:00
#SBATCH --job-name=v15-dynamic-sit
#SBATCH --output=/home/shanghun1404/v15_cad_20261011/dynamic-sit-%j.out
set -euo pipefail
root=/home/shanghun1404/v15_cad_20261011
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
python=/opt/isaacsim-env/bin/python
cases=${1:?case list required}
actor=$root/train_jobs/2341/results/model_15699.pt
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
output=$root/dynamic_sit_jobs/$SLURM_JOB_ID
mkdir -p "$output/source/config" "$output/source/ros2" "$output/source/assets" "$output/source/firmware" "$output/results"
cp "$root/config/"*.py "$root/config/v15_usb_codec_echo.c" "$output/source/config/"
cp -r "$root/config/experiments/v15_cad_20261011/model" "$output/source/config/"
cp -r "$root/ros2/v151_ros2_control" "$root/ros2/v14_ros2_control" "$root/ros2/v15_hardware_bridge" "$output/source/ros2/"
cp -r "$root/assets/quadruped_20261011" "$output/source/assets/"
cp -r "$root/firmware/stm32g474_usb" "$output/source/firmware/"
cp "$root/config/experiments/v15_cad_20261011/dynamic_sit.sbatch" "$output/launcher.sbatch"
gcc -std=c11 -O2 -Wall -Wextra -Werror -I"$output/source/firmware/stm32g474_usb/include" \
 "$output/source/config/v15_usb_codec_echo.c" "$output/source/firmware/stm32g474_usb/src/qd_protocol.c" -lm -o "$output/usb_codec"
export V15_USB_CODEC=$output/usb_codec V15_USB_RESULTS=$output/results
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$root/runtime_dependencies:$output/source/config:$output/source/ros2/v15_hardware_bridge:$output/source/ros2/v151_ros2_control:$output/source/ros2/v14_ros2_control:/home/shanghun1404/v15_large_scale/config:/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
$python -c 'import json,os,sys; json.dump({k:os.environ.get(k) for k in ("RMW_IMPLEMENTATION","ROS_DOMAIN_ID","ROS_DISTRO","SLURM_JOB_ID","SLURM_JOB_QOS","CUDA_VISIBLE_DEVICES")},open(sys.argv[1],"w"),indent=2)' "$output/runtime.json"
find "$output/source" -type f ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' ! -path '*/build/*' -print0 | sort -z | xargs -0 sha256sum > "$output/source.sha256"
sha256sum "$actor" "$student" > "$output/checkpoints.sha256"
exec "$python" "$output/source/config/evaluate_v15_dynamic_sit.py" --headless --enable_cameras \
 --usd-path "$output/source/assets/quadruped_20261011/quadruped_robot.usda" \
 --actor "$actor" --student "$student" --model-path "$output/source/config/model/quadruped_cad.xml" \
 --output-dir "$output/results" --cases "$cases" --seed 42 > "$output/simulator.log" 2>&1
