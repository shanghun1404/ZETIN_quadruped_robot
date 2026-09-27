#!/bin/bash
# Queue the docs/v15_design.md recovery gate evaluations for a V15-R training job.
#   bash queue_v15_recovery_evals.sh <training_job_id>
set -euo pipefail
train=${1:?training job id required}
v15=/home/shanghun1404/v15_large_scale
checkpoint="\$(cat $v15/recovery_jobs/$train/checkpoint.txt)"
for spec in "fallen 42" "fallen 43" "fallen 44" "standing 42"; do
    job=$(sbatch --parsable --job-name=v15-r-eval --dependency=afterok:$train --partition=compute --qos=admin \
        --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --output=$v15/slurm-%j.out \
        --wrap="bash $v15/config/launch_v15_recovery_eval.sbatch $checkpoint main$train $spec")
    echo "$job recovery $spec"
done
