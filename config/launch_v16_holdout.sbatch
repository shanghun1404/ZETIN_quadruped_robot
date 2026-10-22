#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=v16-independent-holdout
#SBATCH --output=/home/shanghun1404/v16_12dof_model/holdout-%j.out
set -euo pipefail
training_job=${1:?Usage: sbatch --dependency=afterok:TRAIN_JOB launch_v16_holdout.sbatch TRAIN_JOB}
case "$training_job" in *[!0-9]*|'') exit 2 ;; esac
root=/home/shanghun1404/v16_12dof_model
seconds=${V16_HOLDOUT_SECONDS:-12}
source_dir=$root/job_sources/$SLURM_JOB_ID
output=$root/evaluation_jobs/$SLURM_JOB_ID
mkdir -p "$source_dir" "$output"
# Freeze the exact controller code and final checkpoint from the training job.
cp "$root/job_sources/$training_job/run_v16_physics.py" "$root/job_sources/$training_job/v16_controller_math.py" "$source_dir/"
cp "$root/job_sources/$training_job/v16_force_qp.py" "$source_dir/"
if [ -f "$root/job_sources/$training_job/v16_wbc_qp.py" ]; then
    cp "$root/job_sources/$training_job/v16_wbc_qp.py" "$source_dir/"
fi
if [ -f "$root/job_sources/$training_job/v16_cem_robust.py" ]; then
    cp "$root/job_sources/$training_job/v16_cem_robust.py" "$source_dir/"
fi
if [ -f "$root/job_sources/$training_job/v16_com_reference_math.py" ]; then
    cp "$root/job_sources/$training_job/v16_com_reference_math.py" "$source_dir/"
fi
if [ -f "$root/job_sources/$training_job/v16_crawl_math.py" ]; then
    cp "$root/job_sources/$training_job/v16_crawl_math.py" "$source_dir/"
fi
cp "$root/physics_jobs/$training_job/checkpoint.json" "$output/training_checkpoint.json"
cp "$root/physics_jobs/$training_job/manifest.json" "$output/training_manifest.json"
cp "$root/config/analyze_v16.py" "$root/config/summarize_v16_holdout.py" "$source_dir/"
printf '%s\n' "$training_job" > "$output/training_job.txt"
sha256sum "$source_dir"/*.py "$output/training_checkpoint.json" > "$output/source_sha256.txt"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=$source_dir:/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11
cd "$source_dir"
/opt/isaacsim-env/bin/python - "$output" "$seconds" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
output=Path(sys.argv[1]); checkpoint=json.loads((output/'training_checkpoint.json').read_text())
evaluation_seconds=float(sys.argv[2])
if not math.isfinite(evaluation_seconds) or not 1<evaluation_seconds<=120:
    raise ValueError('invalid evaluation duration')
expected=checkpoint['controller_code_sha256']
actual={name:hashlib.sha256((Path.cwd()/name).read_bytes()).hexdigest() for name in expected}
if actual!=expected:
    raise ValueError('training checkpoint/controller source SHA mismatch')
if hashlib.sha256((output/'training_manifest.json').read_bytes()).hexdigest()!=checkpoint['manifest_sha256']:
    raise ValueError('training manifest SHA mismatch')
training_seeds={int(checkpoint['training_seed'])+int(row['generation']) for row in checkpoint['history']}
if training_seeds.intersection({1001,1002,1003}):
    raise ValueError('holdout seed appeared during training')
speed=float(checkpoint['speed_command_m_s']); dt=float(checkpoint['control_dt_s'])
decimation=round(dt/.005); axis=checkpoint['walk_axis_world']
if not math.isfinite(speed) or speed<=0 or axis not in ('x','y') or not 1<=decimation<=4 or abs(dt-decimation*.005)>1e-9:
    raise ValueError('invalid saved evaluation conditions')
common=['--speed',str(speed),'--walk-axis',axis,'--control-decimation',str(decimation),
        '--effort-fraction',str(checkpoint['effort_fraction']),
        '--gait-mode',checkpoint.get('gait_mode','periodic'),
        '--stance-point',checkpoint.get('stance_point','com'),
        '--joint-friction',str(checkpoint.get('joint_friction',.3))]
if 'base_acceleration_weight' in checkpoint:
    common += ['--base-acceleration-weight',str(checkpoint['base_acceleration_weight'])]
for name in ('joint_dynamic_friction','joint_viscous_friction'):
    if checkpoint.get(name) is not None:
        common += ['--'+name.replace('_','-'),str(checkpoint[name])]
model=['--contact-projection'] if checkpoint.get('contact_projection',False) else []
model += ['--force-allocator',checkpoint.get('force_allocator','ls')]
if checkpoint.get('force_allocator') == 'wbc':
    model += ['--wbc-iterations',str(checkpoint['wbc_iterations'])]
    if 'wbc_tracking_metric' in checkpoint:
        model += ['--wbc-tracking-metric',checkpoint['wbc_tracking_metric']]
if checkpoint.get('smooth_com_reference',False):
    model += ['--smooth-com-reference']
(output/'common_args.txt').write_text('\n'.join(common)+'\n')
(output/'model_args.txt').write_text('\n'.join(model)+('\n' if model else ''))
(output/'evaluation_conditions.json').write_text(json.dumps({'common_args':common,'model_args':model,
    'controller_source_sha_verified':True,'training_manifest_sha_verified':True,
    'training_seeds':sorted(training_seeds),'holdout_seeds_disjoint':True,
    'evaluation_seconds':evaluation_seconds,
    'source':'training checkpoint; independent evaluation after parameter freeze'},indent=2)+'\n')
PY
mapfile -t common_args < "$output/common_args.txt"
mapfile -t model_args < "$output/model_args.txt"
for seed in 1001 1002 1003; do
    /opt/isaacsim-env/bin/python run_v16_physics.py --headless \
      --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
      --output-root "$output/model_seed$seed" --mode eval --controller model \
      --num-envs 32 --seconds "$seconds" --seed "$seed" --params "$output/training_checkpoint.json" \
      "${common_args[@]}" "${model_args[@]}"
    /opt/isaacsim-env/bin/python analyze_v16.py "$output/model_seed$seed"
done
/opt/isaacsim-env/bin/python run_v16_physics.py --headless \
  --usd-path /home/shanghun1404/v14_leg_collision/assets/robot_v2_v14_leg_collision.usda \
  --output-root "$output/pd_default_seed1001" --mode eval --controller pd \
  --num-envs 32 --seconds "$seconds" --seed 1001 "${common_args[@]}"
/opt/isaacsim-env/bin/python analyze_v16.py "$output/pd_default_seed1001"
/opt/isaacsim-env/bin/python summarize_v16_holdout.py "$output"
