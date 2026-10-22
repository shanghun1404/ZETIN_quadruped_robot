#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13G
#SBATCH --time=02:00:00
#SBATCH --job-name=v15-cad-candidate-suite
#SBATCH --output=/home/shanghun1404/v15_cad_20261011/candidate-suite-%j.out
set -euo pipefail
root=/home/shanghun1404/v15_cad_20261011
train_job=${1:?completed training job ID required}
cases=${2:?comma-separated original suite case names required}
seed=${3:-42}
[[ "$train_job" =~ ^[0-9]+$ ]]
result=$root/train_jobs/$train_job/results
python=/opt/isaacsim-env/bin/python
actor=$("$python" - "$result" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
completed=json.loads((root/'completed.json').read_text())
preflight=json.loads((root/'preflight.json').read_text())
assert completed.get('training_completed') is True
assert preflight.get('passed') is True and preflight.get('nonfinite_steps') == 0
checkpoint=Path(completed['checkpoint'])
assert checkpoint.parent.resolve() == root.resolve()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == completed['checkpoint_sha256']
print(checkpoint)
PY
)
output=$root/suite_jobs/$SLURM_JOB_ID
mkdir -p "$output"
cp "$result/completed.json" "$output/training_completed.json"
cp "$result/preflight.json" "$output/training_preflight.json"
cp "$result/training_contract.json" "$output/training_contract.json"
cp "$root/train_jobs/$train_job/source.sha256" "$output/training_source.sha256"
cp "$root/config/experiments/v15_cad_20261011/finetune_eval.sbatch" "$output/candidate_launcher.sbatch"
exec bash "$root/config/experiments/v15_cad_20261011/suite.sbatch" "$cases" "$actor" "$seed"
