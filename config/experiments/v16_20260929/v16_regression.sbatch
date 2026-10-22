#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=v16-config-regression
#SBATCH --output=/home/shanghun1404/v16_12dof_model/tests-%j.out
set -euo pipefail
root=/home/shanghun1404/v16_12dof_model
snapshot=$root/test_sources/$SLURM_JOB_ID
mkdir -p "$snapshot"
cp -a "$root/test_suite/." "$snapshot/"
cd "$snapshot"
export PYTHONPATH=$snapshot
export V16_MJCF=/home/projects/legacy/robot_v2/quadruped/quadruped.xml
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
sha256sum run_v16_physics.py v16_controller_math.py v16_physical_model.py derive_v16_discrete.py tests/test_v16*.py
/opt/isaacsim-env/bin/python -m pytest tests -q
