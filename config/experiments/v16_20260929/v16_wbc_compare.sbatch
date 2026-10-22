#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=v16-wbc-comparison
#SBATCH --output=/home/shanghun1404/v16_12dof_model/comparison-%j.out
set -euo pipefail
root=/home/shanghun1404/v16_12dof_model
export V16_SOURCE_DIR=$root/config/smooth_robust_v1
output=$root/comparison_jobs/$SLURM_JOB_ID
mkdir -p "$output"
cp "$root/physics_jobs/563/checkpoint.json" "$output/initial_parameters.json"
sha256sum "$output/initial_parameters.json" > "$output/parameter_sha256.txt"
for allocator in qp wbc; do
    export V16_OUTPUT_ROOT=$output/$allocator
    bash "$root/config/launch_v16_physics.sbatch" --mode eval --controller model \
      --num-envs 8 --seconds 16 --speed .05 --seed 71 --walk-axis y \
      --contact-projection --gait-mode prepared_crawl --stance-point ground \
      --force-allocator "$allocator" --base-acceleration-weight 1000 --smooth-com-reference \
      --params "$output/initial_parameters.json"
done
