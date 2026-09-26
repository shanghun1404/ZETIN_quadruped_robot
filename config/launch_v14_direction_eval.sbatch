#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=v14-direction-eval
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out
set -euo pipefail
actor=${1:?actor required}
label=${2:?label required}
scenario=${3:-stairs_up}
seed=${4:-42}
shift 4
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config
output=/home/shanghun1404/v14_leg_collision/results/direction/${label}_${scenario}_seed${seed}
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
/opt/isaacsim-env/bin/python evaluate_v13_torque.py \
  --headless --enable_cameras --variant v13 --num-envs 32 --seed "$seed" \
  --warmup-seconds "$warmup" --seconds 12 --forward-speed "$speed" \
  --scenario "$scenario" "${terrain_args[@]}" --protocol fixed_v2 \
  --effort-guard-fraction 0.99 --enable-self-collisions --filter-leg-terrain --measure-leg-clearance \
  --actor-checkpoint "$actor" \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
  --output "$output.json" --trajectory-output "$output.npz" "$@"
test -s "$output.json"
test -s "$output.npz"
