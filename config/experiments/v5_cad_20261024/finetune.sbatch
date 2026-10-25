#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --job-name=v5-cad-finetune
#SBATCH --output=/home/shanghun1404/v15_cad_20261011/v5-train-%j.out
# Fine-tune the adopted model_15699 on the V5 structure CAD with the round2 recipe (train job 2341):
# mixed terrain, rms_retain, 128 envs, seed 73, std floor 0.1, lr 3e-5, retention weight 5.
# Source is the 2341 snapshot plus train_v5_cad.py; model_15699 is both the start actor and the retention teacher.
set -euo pipefail
root=/home/shanghun1404/v15_cad_20261011
frozen=$root/train_jobs/2341
asset=$root/assets/quadruped_v5_20261024
iterations=${1:-400}
actor=$root/train_jobs/2341/results/model_15699.pt
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
sha256sum -c --quiet "$frozen/source.sha256"
echo "9a440065d0f93835abbf02278f79fff91aea95a18c3bf4020b83bb092f9e6db6  $actor" | sha256sum -c --quiet
job=$root/v5_train_jobs/$SLURM_JOB_ID
mkdir -p "$job/source/assets"
cp -r "$frozen/source/config" "$frozen/source/ros2" "$job/source/"
cp "$root/config/experiments/v5_cad_20261024/train_v5_cad.py" "$job/source/config/"
cp -r "$asset" "$job/source/assets/"
cp "$root/config/experiments/v5_cad_20261024/finetune.sbatch" "$job/launcher.sbatch"
find "$job/source" -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$job/source.sha256"
sha256sum "$actor" "$student" > "$job/checkpoints.sha256"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$job/source/config:$job/source/ros2/v151_ros2_control:$job/source/ros2/v14_ros2_control:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
/opt/isaacsim-env/bin/python -c 'import json,os,sys;json.dump({k:os.environ.get(k) for k in ("SLURM_JOB_ID","SLURM_JOB_QOS","CUDA_VISIBLE_DEVICES","ROS_DOMAIN_ID")},open(sys.argv[1],"w"),indent=2)' "$job/runtime.json"
exec /opt/isaacsim-env/bin/python "$job/source/config/train_v5_cad.py" --headless --enable_cameras \
    --num_envs 128 --iterations "$iterations" --terrain mixed --profile rms_retain --seed 73 \
    --minimum-policy-std .1 --learning-rate .00003 --retention-teacher "$actor" \
    --actor "$actor" --student "$student" --usd "$job/source/assets/quadruped_v5_20261024/quadruped_robot.usda" \
    --output "$job/results" > "$job/training.log" 2>&1
