#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=v15-recovery-usd
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# Build the V15 recovery USD (V14 USD + trunk collider), then drop-test it against the V14 USD.
set -euo pipefail
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
cd $v15/config
mkdir -p $v15/assets $v15/results/probe
usd=$v15/assets/robot_v2_v15_recovery.usda
if [[ ! -e "$usd" ]]; then
    /opt/isaacsim-env/bin/python v15_usd_builder.py --v14-usd $v14/assets/robot_v2_v14_leg_collision.usda \
      --output-usd "$usd" --report-json $v15/results/probe/v15_recovery_usd_build.json
fi
for pair in "v14:$v14/assets/robot_v2_v14_leg_collision.usda" "v15:$usd"; do
    /opt/isaacsim-env/bin/python probe_v15_recovery_usd.py --headless --usd-path "${pair#*:}" \
      --output $v15/results/probe/drop_${pair%%:*}_${SLURM_JOB_ID}.json
done
