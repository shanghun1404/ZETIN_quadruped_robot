#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=13G
#SBATCH --time=01:25:00
#SBATCH --job-name=v15-robust-suite
#SBATCH --output=/home/shanghun1404/v15_robust_20261023/suite-%j.out
# The unchanged nine-case ROS2/Isaac suite on one checkpoint of a robust training job.
# Args: train_job cases checkpoint_file_name [seed]
set -euo pipefail
root=/home/shanghun1404/v15_robust_20261023
train_job=${1:?training job ID required}
cases=${2:?comma-separated original suite case names required}
name=${3:?checkpoint file name required}
seed=${4:-42}
[[ "$train_job" =~ ^[0-9]+$ && "$name" =~ ^model_[0-9]+\.pt$ ]]
result=$root/train_jobs/$train_job/results
actor=$result/$name
python=/opt/isaacsim-env/bin/python
"$python" - "$result" <<'PY'
import json,sys
from pathlib import Path
preflight=json.loads((Path(sys.argv[1])/'preflight.json').read_text())
assert preflight.get('passed') is True and preflight.get('nonfinite_steps') == 0
PY
[[ -f "$actor" ]]
output=$root/suite_jobs/$SLURM_JOB_ID
mkdir -p "$output"
cp "$result/preflight.json" "$output/training_preflight.json"
cp "$result/training_contract.json" "$output/training_contract.json"
cp "$root/train_jobs/$train_job/source.sha256" "$output/training_source.sha256"
cp "$root/config/experiments/v15_robust_20261023/candidate_suite.sbatch" "$output/candidate_launcher.sbatch"
exec bash "$root/config/experiments/v15_robust_20261023/suite.sbatch" "$cases" "$actor" "$seed"
