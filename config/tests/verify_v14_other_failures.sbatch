#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --array=0-2%1
#SBATCH --job-name=v14-cli-fail
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%A_%a.out
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd /home/shanghun1404/v14_leg_collision/config
root=/home/shanghun1404/v14_leg_collision
missing="$root/intentionally_missing_input"
output="$root/results/expected_failure_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
case "$SLURM_ARRAY_TASK_ID" in
0)
  /opt/isaacsim-env/bin/python train_v14_leg_collision.py --headless \
    --actor-checkpoint "$missing" \
    --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
    --usd-path "$root/assets/robot_v2_v14_leg_collision.usda" --output-root "$output"
  status=$?
  ;;
1)
  /opt/isaacsim-env/bin/python probe_v14_leg_physics.py --headless \
    --usd-path "$missing" --mode stand --output "$output"
  status=$?
  ;;
2)
  /opt/isaacsim-env/bin/python v14_usd_builder.py \
    --source-usd "$missing" --output-usd "$output.usda" --report-json "$output"
  status=$?
  ;;
esac
printf 'expected_failure_exit_code=%s\n' "$status"
if [[ "$status" != 1 ]] || [[ -e "$output" ]] || [[ -e "$output.usda" ]]; then
  echo "Failure exit contract did not hold" >&2
  exit 1
fi
echo "Missing input correctly failed with exit 1 and no success artifact"
