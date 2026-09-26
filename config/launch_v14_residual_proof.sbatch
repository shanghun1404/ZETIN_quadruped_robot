#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=v14-zero-proof
#SBATCH --output=/home/shanghun1404/v14_leg_collision/slurm-%j.out
set -euo pipefail
baseline=${1:?baseline JSON required}
candidate=${2:?zero candidate JSON required}
output=${3:?proof JSON required}
test ! -e "$output"
cd /home/shanghun1404/v14_leg_collision/config
/opt/isaacsim-env/bin/python v14_residual_proof.py "$baseline" "$candidate" "$output"
test -s "$output"
