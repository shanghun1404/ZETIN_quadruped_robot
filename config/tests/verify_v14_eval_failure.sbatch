#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=v14-fail-exit
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
output=/home/shanghun1404/v14_leg_collision/results/expected_failure_${SLURM_JOB_ID}.json
/opt/isaacsim-env/bin/python /home/shanghun1404/v14_leg_collision/config/evaluate_v13_torque.py \
  --headless --variant v13 \
  --actor-checkpoint /home/shanghun1404/v14_leg_collision/intentionally_missing_checkpoint.pt \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
  --output "$output"
status=$?
printf 'expected_failure_exit_code=%s\n' "$status"
if [[ "$status" != 1 ]] || [[ -e "$output" ]]; then
  echo "Failure exit contract did not hold" >&2
  exit 1
fi
echo "Missing checkpoint correctly failed with exit 1 and no success report"
