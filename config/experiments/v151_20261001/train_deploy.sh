#!/bin/bash
# Deploy only; intentionally does not submit Slurm jobs.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
root=/home/shanghun1404/v151_ros2
ssh -o ClearAllForwardings=yes ws1 "mkdir -p $root/config/experiments/v151_20261001 $root/config/tests"
scp -o ClearAllForwardings=yes "$repo/config/train_v151_locomotion.py" "$repo/config/v151_training_math.py" "ws1:$root/config/"
scp -o ClearAllForwardings=yes "$repo/config/tests/test_v151_training_math.py" "ws1:$root/config/tests/"
scp -o ClearAllForwardings=yes "$repo/config/experiments/v151_20261001/train_v151.sbatch" "$repo/config/experiments/v151_20261001/train_sources.sha256" "ws1:$root/config/experiments/v151_20261001/"
ssh -o ClearAllForwardings=yes ws1 "cd $root/config && sha256sum --check experiments/v151_20261001/train_sources.sha256"
