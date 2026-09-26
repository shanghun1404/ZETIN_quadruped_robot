#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=v14-residual
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out
set -euo pipefail
mode=${1:?zero, train or eval required}
shift
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config
base=/home/shanghun1404/v14_leg_collision/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt
if [[ "$mode" == eval ]]; then
    checkpoint=${1:?checkpoint required}
    label=${2:?label required}
    scenario=${3:?scenario required}
    seed=${4:?seed required}
    exec bash launch_v14_direction_eval.sbatch "$base" "$label" "$scenario" "$seed" --heading-control --residual-checkpoint "$checkpoint"
fi
output=/home/shanghun1404/v14_leg_collision/residual_jobs/${SLURM_JOB_ID}
extra=()
if [[ "$mode" == zero ]]; then
    extra=(--zero-only)
elif [[ "$mode" == train ]]; then
    proof=${1:?zero proof JSON required}
    iterations=${2:-300}
    extra=(--zero-proof "$proof" --iterations "$iterations")
else
    echo "unknown mode: $mode" >&2
    exit 2
fi
/opt/isaacsim-env/bin/python train_v14_residual.py \
    --headless --enable_cameras --num_envs 32 --seed 46 \
    --actor-checkpoint "$base" \
    --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
    --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
    --output-dir "$output" "${extra[@]}"
test -s "$output/model_zero.pt"
test -s "$output/verification.json"
if [[ "$mode" == train ]]; then
    test -s "$output/model_$((iterations-1)).pt"
fi
