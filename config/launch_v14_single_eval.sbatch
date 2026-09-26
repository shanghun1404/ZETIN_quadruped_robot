#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=v14-eval-one
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out

set -euo pipefail
actor=${1:?actor checkpoint is required}
label=${2:?result label is required}
scenario=${3:?scenario is required}
seed=${4:-42}
test -s "$actor"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config

warmup=0
speed=0.5
terrain_args=()
if [[ "$scenario" == flat ]]; then
  warmup=2
  speed=0.4
elif [[ "$scenario" == rough ]]; then
  terrain_args=(--terrain-level 4)
elif [[ "$scenario" == stairs_up || "$scenario" == stairs_down ]]; then
  terrain_args=(--terrain-level 5)
else
  echo "unsupported scenario: $scenario" >&2
  exit 2
fi

results=/home/shanghun1404/v14_leg_collision/results/actor_gate
mkdir -p "$results"
output="$results/${label}_${scenario}_seed${seed}.json"
if [[ -e "$output" ]]; then
  echo "result already exists; use a new label: $output" >&2
  exit 2
fi
/opt/isaacsim-env/bin/python evaluate_v13_torque.py \
  --headless --enable_cameras --variant v13 --num-envs 32 --seed "$seed" \
  --warmup-seconds "$warmup" --seconds 12 --forward-speed "$speed" \
  --scenario "$scenario" "${terrain_args[@]}" \
  --effort-guard-fraction 0.99 --enable-self-collisions \
  --filter-leg-terrain --measure-leg-clearance \
  --actor-checkpoint "$actor" \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
  --output "$output"
# Also catch an early Kit/import failure before Python's main exception handler.
test -s "$output"
