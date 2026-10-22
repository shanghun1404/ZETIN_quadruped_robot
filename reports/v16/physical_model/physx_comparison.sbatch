#!/bin/bash
#SBATCH --job-name=v16-physx-compare
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/shanghun1404/v16_12dof_model/model/reports/physx_comparison-%j.out
#SBATCH --error=/home/shanghun1404/v16_12dof_model/model/reports/physx_comparison-%j.err
set -euo pipefail
cd /home/shanghun1404/v16_12dof_model/model
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
probe_directory=${1:?Provide the completed probe output directory}
test -f "$probe_directory/physx_nominal.npz"
test -f "$probe_directory/manifest.json"
/opt/isaacsim-env/bin/python config/experiments/v16_20260929/v16_compare_physx.py \
  --probe-dir "$probe_directory" \
  --mjcf /home/projects/legacy/robot_v2/quadruped/quadruped.xml \
  --output-prefix "reports/physx_comparison_${SLURM_JOB_ID}" \
  --inertia-frame actor --physx-armature-in-mass "${2:-unknown}"
