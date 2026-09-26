#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-11%2
#SBATCH --job-name=v14-validation
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%A_%a.out

# One Isaac Sim process per allocation avoids sequential renderer startup failures.
# Usage: sbatch launch_v14_validation.sbatch ACTOR_CHECKPOINT UNIQUE_LABEL
set -euo pipefail
actor=${1:?selected V14 actor checkpoint is required}
label=${2:?unique result label is required}
task=${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}
if [[ ! "$task" =~ ^([0-9]|1[01])$ ]]; then
  echo "array task must be in [0, 11]" >&2
  exit 2
fi
scenarios=(flat rough stairs_up stairs_down)
seed=$((42 + task / 4))
scenario=${scenarios[task % 4]}
exec bash /home/shanghun1404/v14_leg_collision/config/launch_v14_single_eval.sbatch \
  "$actor" "$label" "$scenario" "$seed"
