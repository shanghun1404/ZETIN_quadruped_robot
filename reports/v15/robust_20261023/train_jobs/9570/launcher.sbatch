#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15900M
#SBATCH --time=01:25:00
#SBATCH --job-name=v15-robust-train
#SBATCH --output=/home/shanghun1404/v15_robust_20261023/train-%j.out
# Robust fine-tune of the Jetson-bundled V15 actor (model_15699, SHA 9a440065...).
# Args: iterations ramp_iterations num_envs [extra train_v15_cad.py args, e.g. --robust-ramp-offset-iterations N].
# Segments stay under 1:25 so backfill can run them beside other users' 1:30 jobs.
# Env RESUME_JOB continues from that completed segment's final checkpoint and ramp position;
# the retention teacher stays model_15699.
set -euo pipefail
root=/home/shanghun1404/v15_robust_20261023
job=$root/train_jobs/$SLURM_JOB_ID
iterations=${1:?iterations required}
ramp=${2:?ramp iterations required}
num_envs=${3:?num_envs required}
extra=("${@:4}")
teacher=/home/shanghun1404/v15_cad_20261011/train_jobs/2341/results/model_15699.pt
actor=$teacher
resume_args=()
if [[ -n "${RESUME_JOB:-}" ]]; then
    [[ "$RESUME_JOB" =~ ^[0-9]+$ ]]
    read -r actor offset < <(/opt/isaacsim-env/bin/python - "$root/train_jobs/$RESUME_JOB/results" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
done=json.loads((root/'completed.json').read_text())
assert done['training_completed'] is True
checkpoint=Path(done['checkpoint'])
assert checkpoint.parent.resolve()==root.resolve()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==done['checkpoint_sha256']
print(checkpoint,done['last_iteration']-15699)
PY
    )
    # Keep the trained std (no re-boost) and vary the seed per segment.
    resume_args=(--robust-ramp-offset-iterations "$offset" --minimum-policy-std 0 --seed $((73+offset)))
fi
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
mkdir -p "$job/source/config" "$job/source/assets" "$job/source/ros2"
cp "$root/config/"*.py "$job/source/config/"
cp -r "$root/config/experiments/v15_cad_20261011/model" "$job/source/config/"
cp -r "$root/assets/quadruped_20261011" "$job/source/assets/"
cp -r "$root/ros2/v151_ros2_control" "$root/ros2/v14_ros2_control" "$job/source/ros2/"
cp "$root/config/experiments/v15_robust_20261023/train.sbatch" "$job/launcher.sbatch"
find "$job/source" -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$job/source.sha256"
sha256sum "$actor" "$teacher" "$student" > "$job/checkpoints.sha256"
echo "9a440065d0f93835abbf02278f79fff91aea95a18c3bf4020b83bb092f9e6db6  $teacher" | sha256sum -c -
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$job/source/config:$job/source/ros2/v151_ros2_control:$job/source/ros2/v14_ros2_control:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
/opt/isaacsim-env/bin/python -c 'import json,os,sys;json.dump({k:os.environ.get(k) for k in ("SLURM_JOB_ID","SLURM_JOB_QOS","CUDA_VISIBLE_DEVICES","ROS_DOMAIN_ID")},open(sys.argv[1],"w"),indent=2)' "$job/runtime.json"
# Same profile, seed, lr and std floor as job2341; retention anchors to the adopted actor itself at weight 1.
exec /opt/isaacsim-env/bin/python "$job/source/config/train_v15_cad.py" --headless --enable_cameras \
    --num_envs "$num_envs" --iterations "$iterations" --terrain mixed --profile rms_retain --seed 73 \
    --minimum-policy-std .10 --learning-rate .00003 \
    --retention-teacher "$teacher" --retention-weight 1.0 \
    --robust --robust-ramp-iterations "$ramp" \
    --actor "$actor" --student "$student" --usd "$job/source/assets/quadruped_20261011/quadruped_robot.usda" \
    --output "$job/results" "${resume_args[@]}" "${extra[@]}" > "$job/training.log" 2>&1
