#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=motor-v16-wbc
#SBATCH --output=/home/shanghun1404/motor_audit_20260930/v16-%j.log
set -euo pipefail
root=/home/shanghun1404/motor_audit_20260930
v16=/home/shanghun1404/v16_12dof_model
out=$root/v16_$SLURM_JOB_ID
mkdir -p "$out/source"
cp "$v16/job_sources/577/"*.py "$out/source/"
cp "$root/source/run_v16_physics.py" "$out/source/"
cp "$root/source/v16_audit.sbatch" "$out/source/"
cp "$v16/evaluation_jobs/579/training_checkpoint.json" "$out/"
cp "$v16/evaluation_jobs/579/common_args.txt" "$v16/evaluation_jobs/579/model_args.txt" "$out/"
sha256sum "$out/source/"*.py "$out/training_checkpoint.json" > "$out/source.sha256"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH=$out/source:/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd "$out/source"
mapfile -t common_args < "$out/common_args.txt"
mapfile -t model_args < "$out/model_args.txt"
for terrain in flat; do
  timeout --signal=TERM --kill-after=30s 25m /opt/isaacsim-env/bin/python run_v16_physics.py \
    --headless --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
    --output-root "$out/V16_${terrain}_seed1001" --mode eval --controller model \
    --num-envs 8 --seconds 32 --seed 1001 --params "$out/training_checkpoint.json" \
    --audit-terrain "$terrain" "${common_args[@]}" "${model_args[@]}" \
    > "$out/V16_${terrain}.log" 2>&1
  test -s "$out/V16_${terrain}_seed1001/motor_trace_seed1001.npz"
done
sha256sum "$out"/V16_*/*.npz "$out"/V16_*/*.json > "$out/data.sha256"
