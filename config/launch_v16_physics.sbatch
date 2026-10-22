#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=07:00:00
#SBATCH --job-name=v16-physical-control
#SBATCH --output=/home/shanghun1404/v16_12dof_model/slurm-%j.out
set -euo pipefail
root=/home/shanghun1404/v16_12dof_model
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
source_dir=$root/job_sources/$SLURM_JOB_ID
input_sources=${V16_SOURCE_DIR:-$root/config}
output=${V16_OUTPUT_ROOT:-$root/physics_jobs/$SLURM_JOB_ID}
mkdir -p "$source_dir"
cp "$input_sources/run_v16_physics.py" "$input_sources/v16_controller_math.py" "$source_dir/"
cp "$input_sources/v16_force_qp.py" "$source_dir/"
if [ -f "$input_sources/v16_wbc_qp.py" ]; then
    cp "$input_sources/v16_wbc_qp.py" "$source_dir/"
fi
if [ -f "$input_sources/v16_cem_robust.py" ]; then
    cp "$input_sources/v16_cem_robust.py" "$source_dir/"
fi
if [ -f "$input_sources/v16_com_reference_math.py" ]; then
    cp "$input_sources/v16_com_reference_math.py" "$source_dir/"
fi
if [ -f "$input_sources/v16_crawl_math.py" ]; then
    cp "$input_sources/v16_crawl_math.py" "$source_dir/"
fi
cp "$root/config/analyze_v16.py" "$source_dir/"
export PYTHONPATH=$source_dir:/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd "$source_dir"
/opt/isaacsim-env/bin/python run_v16_physics.py --headless \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
  --output-root "$output" "$@"
/opt/isaacsim-env/bin/python analyze_v16.py "$output"
