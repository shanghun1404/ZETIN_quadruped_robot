#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=v16-trained-video
#SBATCH --output=/home/shanghun1404/v16_12dof_model/video-%j.out
set -euo pipefail
root=/home/shanghun1404/v16_12dof_model
training_job=${1:?training job required}
case "$training_job" in *[!0-9]*|'') exit 2 ;; esac
export V16_SOURCE_DIR=$root/job_sources/$training_job
export V16_OUTPUT_ROOT=$root/video_jobs/$SLURM_JOB_ID
# Plant creates a fresh output directory and rejects any pre-existing one.
# Prepare inputs separately, then publish their provenance after the rollout.
input_dir=$root/video_inputs/$SLURM_JOB_ID
mkdir -p "$input_dir"
cp "$root/physics_jobs/$training_job/checkpoint.json" "$input_dir/training_checkpoint.json"
/opt/isaacsim-env/bin/python - "$input_dir" "$V16_SOURCE_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
output=Path(sys.argv[1]); source=Path(sys.argv[2])
checkpoint=json.loads((output/'training_checkpoint.json').read_text())
if any(hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest
       for name,digest in checkpoint['controller_code_sha256'].items()):
    raise ValueError('video controller source/checkpoint mismatch')
arguments=['--speed',str(checkpoint['speed_command_m_s']),
    '--walk-axis',checkpoint['walk_axis_world'],
    '--control-decimation',str(round(checkpoint['control_dt_s']/.005)),
    '--effort-fraction',str(checkpoint['effort_fraction']),
    '--gait-mode',checkpoint['gait_mode'],'--stance-point',checkpoint['stance_point'],
    '--joint-friction',str(checkpoint['joint_friction']),
    '--force-allocator',checkpoint['force_allocator']]
if 'base_acceleration_weight' in checkpoint:
    arguments+=['--base-acceleration-weight',str(checkpoint['base_acceleration_weight'])]
for name in ('joint_dynamic_friction','joint_viscous_friction'):
    if checkpoint.get(name) is not None:
        arguments+=['--'+name.replace('_','-'),str(checkpoint[name])]
if checkpoint.get('contact_projection'):
    arguments+=['--contact-projection']
if checkpoint.get('smooth_com_reference'):
    arguments+=['--smooth-com-reference']
if checkpoint['force_allocator']=='wbc':
    arguments+=['--wbc-iterations',str(checkpoint['wbc_iterations'])]
    if 'wbc_tracking_metric' in checkpoint:
        arguments+=['--wbc-tracking-metric',checkpoint['wbc_tracking_metric']]
(output/'controller_args.txt').write_text('\n'.join(arguments)+'\n')
PY
mapfile -t controller_args < "$input_dir/controller_args.txt"
# Same frozen trained controller; observer RGB renders do not advance physics.
bash "$root/config/launch_v16_physics.sbatch" --mode eval --controller model \
  --num-envs 1 --seconds 24 --seed 1001 "${controller_args[@]}" \
  --params "$input_dir/training_checkpoint.json" --video --enable_cameras
cp "$input_dir/training_checkpoint.json" "$input_dir/controller_args.txt" "$V16_OUTPUT_ROOT/"
