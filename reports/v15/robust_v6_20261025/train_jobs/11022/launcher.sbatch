#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15900M
#SBATCH --time=01:25:00
#SBATCH --job-name=v6-robust-train
#SBATCH --output=/home/shanghun1404/v15_cad_20261011/v6-robust-train-%j.out
# Continuation segments of the robust V6 fine-tune chain started by train.sbatch (job 10854): same training
# command, but each segment runs min(cap, target - resumed iteration) updates so it completes inside 1:25,
# and the segment that reaches the target links its final checkpoint to $root/v6_robust_train_jobs/<link>.
# Args: target_iteration ramp_iterations num_envs cap link_name [extra train_v6_robust.py args].
# Env RESUME_JOB (required) is the previous segment; submit with --dependency=afterany.
# Once the link exists, later segments exit without training. The retention teacher stays model_15699.
set -euo pipefail
root=/home/shanghun1404/v15_cad_20261011
frozen=/home/shanghun1404/v15_robust_20261023/train_jobs/9941
asset=$root/assets/quadruped_v6_20261025
experiment=$root/config/experiments/v6_robust_20261025
job=$root/v6_robust_train_jobs/$SLURM_JOB_ID
target=${1:?target iteration required}
ramp=${2:?ramp iterations required}
num_envs=${3:?num_envs required}
cap=${4:?per-segment iteration cap required}
link=$root/v6_robust_train_jobs/${5:?link name required}
extra=("${@:6}")
if [[ -e "$link" ]]; then echo "target already reached: $(readlink -f "$link")"; exit 0; fi
teacher=$root/train_jobs/2341/results/model_15699.pt
actor=$teacher
seed=73
resume_args=()
if [[ -n "${RESUME_JOB:?RESUME_JOB required}" ]]; then
    [[ "$RESUME_JOB" =~ ^[0-9]+$ ]]
    read -r actor offset < <(/opt/isaacsim-env/bin/python - "$root/v6_robust_train_jobs/$RESUME_JOB/results" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch
root=Path(sys.argv[1])
done=root/'completed.json'
if done.is_file():
    done=json.loads(done.read_text())
    assert done['training_completed'] is True
    checkpoint=Path(done['checkpoint'])
    assert checkpoint.parent.resolve()==root.resolve()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==done['checkpoint_sha256']
    iteration=done['last_iteration']
else:
    # The segment hit its time limit: use its newest checkpoint that loads.
    for checkpoint in sorted(root.glob('model_*.pt'),key=lambda path:int(path.stem[6:]),reverse=True):
        try:torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
        except Exception:continue
        break
    else:raise SystemExit('no loadable checkpoint')
    iteration=int(checkpoint.stem[6:])
print(checkpoint,iteration-15699)
PY
    )
    # Keep the trained std (no re-boost) and vary the seed per segment.
    seed=$((73+offset))
    resume_args=(--robust-ramp-offset-iterations "$offset" --minimum-policy-std 0)
fi
remaining=$((target-15699-offset))
(( remaining > 0 )) || { echo "resumed iteration $((15699+offset)) is past target $target"; exit 1; }
iterations=$(( remaining < cap ? remaining : cap ))
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
(cd "$frozen" && sha256sum -c --quiet source.sha256)
echo "9a440065d0f93835abbf02278f79fff91aea95a18c3bf4020b83bb092f9e6db6  $teacher" | sha256sum -c --quiet
mkdir -p "$job/source/assets"
rsync -a --exclude __pycache__ "$frozen/source/config" "$frozen/source/ros2" "$job/source/"
cp "$experiment/train_v6_robust.py" "$job/source/config/"
cp -r "$asset" "$job/source/assets/"
cp "$experiment/train_target.sbatch" "$job/launcher.sbatch"
find "$job/source" -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$job/source.sha256"
sha256sum "$actor" "$teacher" "$student" > "$job/checkpoints.sha256"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export PYTHONPATH=$job/source/config:$job/source/ros2/v151_ros2_control:$job/source/ros2/v14_ros2_control:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
/opt/isaacsim-env/bin/python -c 'import json,os,sys;json.dump({k:os.environ.get(k) for k in ("SLURM_JOB_ID","SLURM_JOB_QOS","CUDA_VISIBLE_DEVICES","ROS_DOMAIN_ID")},open(sys.argv[1],"w"),indent=2)' "$job/runtime.json"
# Same profile, lr and std floor as job2341 and robust 9941; retention anchors to model_15699 at weight 5.
# Kit start-up intermittently fails with a viewport-camera null prim, so a failure before preflight retries once, with the next seed.
for attempt in 1 2; do
if /opt/isaacsim-env/bin/python "$job/source/config/train_v6_robust.py" --headless --enable_cameras \
    --num_envs "$num_envs" --iterations "$iterations" --terrain mixed --profile rms_retain --seed "$seed" \
    --minimum-policy-std .10 --learning-rate .00003 \
    --retention-teacher "$teacher" --retention-weight 5.0 \
    --robust --robust-profile motor --robust-ramp-iterations "$ramp" \
    --actor "$actor" --student "$student" --usd "$job/source/assets/quadruped_v6_20261025/quadruped_robot.usda" \
    --output "$job/results" "${resume_args[@]}" "${extra[@]}" > "$job/training.log" 2>&1; then
    final=$job/results/model_$target.pt
    if python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(not(d["training_completed"] is True and d["last_iteration"]==int(sys.argv[2]) and d["checkpoint"]==sys.argv[3]))' \
        "$job/results/completed.json" "$target" "$final"; then ln -s "$final" "$link"; fi
    exit 0
fi
[[ $attempt == 1 && ! -f "$job/results/preflight.json" ]] || exit 1
mv "$job/training.log" "$job/training_startup_failure_seed$seed.log"
rm -rf "$job/results"
seed=$((seed+1))
done
