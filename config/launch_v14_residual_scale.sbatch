#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=v14-residual-scale
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out
set -euo pipefail
checkpoint=${1:?residual checkpoint required}
num_envs=${2:-128}
iterations=${3:-3000}
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config
output=/home/shanghun1404/v14_leg_collision/residual_jobs/${SLURM_JOB_ID}
/opt/isaacsim-env/bin/python train_v14_residual.py \
    --headless --enable_cameras --num_envs "$num_envs" --seed 46 \
    --iterations "$iterations" --save-interval 100 --resume-residual "$checkpoint" \
    --zero-proof /home/shanghun1404/v14_leg_collision/residual_jobs/zero_proof171.json \
    --actor-checkpoint /home/shanghun1404/v14_leg_collision/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt \
    --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
    --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
    --output-dir "$output"
test -s "$output/model_resume.pt"
/opt/isaacsim-env/bin/python - "$output" "$iterations" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
manifest = json.loads((root/'v14_residual_manifest.json').read_text())
verification = json.loads((root/'verification.json').read_text())
assert verification['updates_completed'] == int(sys.argv[2])
assert verification['last_iteration'] == manifest['last_iteration']
assert (root/f"model_{manifest['last_iteration']}.pt").is_file()
assert verification['base_unchanged'] and verification['student_unchanged']
PY
