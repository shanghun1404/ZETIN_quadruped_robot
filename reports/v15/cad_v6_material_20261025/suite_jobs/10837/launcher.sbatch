#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13G
#SBATCH --time=00:45:00
#SBATCH --job-name=v6-material-suite
#SBATCH --output=/home/shanghun1404/v15_cad_20261011/v6-material-suite-%j.out
# V6 mass variants with metal leg critical parts (make_mass_variants.py: com_pro, hip yokes, knee, knee rear cap scaled by rho/1.24)
# under the V6 fine-tuned model_16099 (train job 10806) by default. Only the robot USD differs from v6_cad_20261025/suite.sbatch.
set -euo pipefail
root=/home/shanghun1404/v15_cad_20261011
v14=/home/shanghun1404/v14_leg_collision
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
python=/opt/isaacsim-env/bin/python
variant=${1:?al7075 or tc4 required}
cases=${2:?comma-separated case names required}
seed=${3:-42}
actor=${4:-$root/v6_train_jobs/10806/results/model_16099.pt}
frozen=$root/suite_jobs/2343
experiment=$root/config/experiments/v6_material_20261025
asset=$root/assets/quadruped_v6_${variant}_20261025
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
sha256sum -c --quiet "$frozen/source.sha256" "$frozen/checkpoints.sha256" "$experiment/assets.sha256"
output=$root/v6_material_suite_jobs/$SLURM_JOB_ID
mkdir -p "$output/source/assets"
cp -r "$frozen/source/config" "$frozen/source/ros2" "$output/source/"
cp -r "$asset" "$output/source/assets/"
cp "$experiment/suite.sbatch" "$output/launcher.sbatch"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$output/source/config:$output/source/ros2/v151_ros2_control:$output/source/ros2/v14_ros2_control:/home/shanghun1404/v15_large_scale/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
$python -c 'import json,os,sys; json.dump({k:os.environ.get(k) for k in ("RMW_IMPLEMENTATION","ROS_DOMAIN_ID","ROS_DISTRO","SLURM_JOB_ID","SLURM_JOB_QOS","CUDA_VISIBLE_DEVICES")},open(sys.argv[1],"w"),indent=2)' "$output/runtime.json"
find "$output/source" -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$output/source.sha256"
sha256sum "$actor" "$student" > "$output/checkpoints.sha256"
exec "$python" "$output/source/config/evaluate_v15_cad_suite.py" --headless --enable_cameras \
 --usd-path "$output/source/assets/quadruped_v6_${variant}_20261025/quadruped_robot.usda" \
 --actor "$actor" --student "$student" --model-path "$output/source/config/model/quadruped_cad.xml" \
 --output-dir "$output/results" --cases "$cases" --seed "$seed" > "$output/simulator.log" 2>&1
