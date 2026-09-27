#!/bin/bash
# Queue the docs/v15_design.md locomotion gate evaluations for a V15-L training job.
#   bash queue_v15_locomotion_gates.sh <training_job_id>
# Each evaluation starts after the training job succeeds and reads its checkpoint.txt.
set -euo pipefail
train=${1:?training job id required}
v15=/home/shanghun1404/v15_large_scale
checkpoint="\$(cat $v15/locomotion_jobs/$train/checkpoint.txt)"
common=(--dependency=afterok:$train --partition=compute --qos=admin --gres=gpu:1
        --cpus-per-task=4 --mem=16G --time=00:40:00 --output=$v15/slurm-%j.out)
evals=("flat 42" "flat 43" "flat 44"
       "flat 42 --heading-control" "stairs_up 42 --heading-control" "stairs_up 43 --heading-control"
       "stairs_up 44 --heading-control" "stairs_down 42 --heading-control" "rough 42 --heading-control")
for spec in "${evals[@]}"; do
    label=main$train
    [[ "$spec" == *--heading-control ]] && label=main${train}_heading
    job=$(sbatch --parsable --job-name=v15-l-gate "${common[@]}" \
        --wrap="bash $v15/config/launch_v15_eval.sbatch $checkpoint $label $spec")
    echo "$job $label $spec"
done
for seed in 42 43 44; do
    job=$(sbatch --parsable --job-name=v15-l-ros2 "${common[@]}" \
        --wrap="bash /home/shanghun1404/v14_leg_collision/config/launch_v14_ros2_eval.sbatch ros2 v15main$train flat $seed $checkpoint")
    echo "$job ros2 flat $seed"
done
