#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=v151-dds-probe
#SBATCH --output=/home/shanghun1404/v151_ros2/slurm-dds-%j.out
set -euo pipefail
root=/home/shanghun1404/v151_ros2
source "$root/config/experiments/v151_20261001/cpu_tests.sh"
export ROS_DOMAIN_ID=$((SLURM_JOB_ID % 90 + 10))
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
output=$root/transport_jobs/$SLURM_JOB_ID
mkdir -p "$output"
for middleware in rmw_fastrtps_cpp rmw_cyclonedds_cpp; do
    export RMW_IMPLEMENTATION=$middleware
    /opt/isaacsim-env/bin/python "$root/startup_probe.py" "$output/$middleware.controller.log" --kit > "$output/$middleware.log" 2>&1 || true
done
