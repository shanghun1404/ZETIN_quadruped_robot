#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=01:25:00
#SBATCH --job-name=v15-robust-eval
#SBATCH --output=/home/shanghun1404/v15_robust_20261023/eval-%j.out
# Args: steps num_envs scenarios(comma list or "all") label=checkpoint [label=checkpoint ...]
set -euo pipefail
root=/home/shanghun1404/v15_robust_20261023
output=$root/eval_jobs/$SLURM_JOB_ID
steps=${1:?steps required}
num_envs=${2:?num_envs required}
scenarios=${3:?scenarios required}
[[ "$scenarios" == all ]] && scenarios=
actors=()
for item in "${@:4}"; do actors+=(--actor "$item"); done
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
mkdir -p "$output/source/config" "$output/source/assets"
cp "$root/config/"*.py "$output/source/config/"
cp -r "$root/config/experiments/v15_cad_20261011/model" "$output/source/config/"
cp -r "$root/assets/quadruped_20261011" "$output/source/assets/"
cp -r "$root/ros2/v151_ros2_control" "$root/ros2/v14_ros2_control" "$output/source/"
cp "$root/config/experiments/v15_robust_20261023/robust_eval.sbatch" "$output/launcher.sbatch"
find "$output/source" -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$output/source.sha256"
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$output/source/config:$output/source/v151_ros2_control:$output/source/v14_ros2_control:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
# Kit start-up intermittently fails (viewport camera null prim); retry once if no scenario finished.
for attempt in 1 2; do
if /opt/isaacsim-env/bin/python "$output/source/config/evaluate_v15_cad_robust.py" --headless --enable_cameras \
    --num_envs "$num_envs" --steps "$steps" --scenarios "$scenarios" "${actors[@]}" --student "$student" \
    --usd "$output/source/assets/quadruped_20261011/quadruped_robot.usda" --output "$output/results" > "$output/eval.log" 2>&1; then exit 0; fi
[[ $attempt == 1 && ! -f "$output/results/robust_eval.json" ]] || exit 1
mv "$output/eval.log" "$output/eval_startup_failure.log"
rm -rf "$output/results"
done
