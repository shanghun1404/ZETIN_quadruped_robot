#!/bin/bash
#SBATCH --job-name=v16-model-verify
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/home/shanghun1404/v16_12dof_model/model/reports/slurm-%j.out
#SBATCH --error=/home/shanghun1404/v16_12dof_model/model/reports/slurm-%j.err
set -euo pipefail
cd /home/shanghun1404/v16_12dof_model/model
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export V16_MJCF=/home/projects/legacy/robot_v2/quadruped/quadruped.xml
/opt/isaacsim-env/bin/python -m unittest discover -s config/tests -p test_v16_model.py -v
if [ -f config/derive_v16_discrete.py ]; then
  /opt/isaacsim-env/bin/python config/derive_v16_discrete.py --mjcf "$V16_MJCF" --output reports
fi
if [ -d mujoco_deps ]; then
  PYTHONPATH="$PWD/mujoco_deps" /opt/isaacsim-env/bin/python config/experiments/v16_20260929/v16_model_mujoco_oracle.py --mjcf "$V16_MJCF" --output reports/v16_model_mujoco_oracle.json
fi
