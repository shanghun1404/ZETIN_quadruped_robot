#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=v15-recovery-eval
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
#   sbatch launch_v15_recovery_eval.sbatch <checkpoint> <label> <fallen|standing> <seed>
set -euo pipefail
checkpoint=${1:?checkpoint required}
label=${2:?label required}
start=${3:?start required}
seed=${4:?seed required}
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
cd $v15/config
output=$v15/results/recovery/${label}_${start}_seed${seed}.json
test ! -e "$output"
/opt/isaacsim-env/bin/python evaluate_v15_recovery.py --headless --checkpoint "$checkpoint" \
  --usd-path $v15/assets/robot_v2_v15_recovery.usda --start "$start" --seed "$seed" --output "$output"
test -s "$output"
