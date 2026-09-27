#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=v15-eval
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# 32-robot fixed_v2 evaluation on the V15 plant (PhysX joint-drive PD, in-process LPF/guard).
#   sbatch launch_v15_eval.sbatch <actor> <label> <flat|stairs_up|stairs_down|rough> <seed> [--heading-control]
set -euo pipefail
actor=${1:?actor required}
label=${2:?label required}
scenario=${3:?scenario required}
seed=${4:?seed required}
shift 4
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
cd $v15/config
output=$v15/results/${label}_${scenario}_seed${seed}
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
  --joint-drive physx --effort-guard-fraction 0.99 \
  --enable-self-collisions --filter-leg-terrain --measure-leg-clearance \
  --actor-checkpoint "$actor" \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path $v14/assets/robot_v2_v14_leg_collision.usda \
  --output "$output.json" --trajectory-output "$output.npz" "$@"
test -s "$output.json"
test -s "$output.npz"
