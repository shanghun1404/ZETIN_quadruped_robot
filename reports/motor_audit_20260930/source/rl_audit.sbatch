#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=motor-v13-v15
#SBATCH --output=/home/shanghun1404/motor_audit_20260930/rl-%j.log
set -euo pipefail
root=/home/shanghun1404/motor_audit_20260930
v14=/home/shanghun1404/v14_leg_collision
v15=/home/shanghun1404/v15_large_scale
legacy=/home/projects/legacy/quadruped_training_smooth_v11
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:$legacy:$v14/ros2/v14_ros2_control
out=$root/rl_$SLURM_JOB_ID
mkdir -p "$out"
cp "$root/source/rl_audit.sbatch" "$root/source/evaluate_original.py" "$out/"
cd "$v15/config"
for version in V13 V14 V15; do
  options=()
  if [[ "$version" == V13 ]]; then
    actor=$legacy/logs/rsl_rl/guide_v11_scanner_terrain_robot_v2/2026-08-12_07-19-56_robot_v2_scanner_terrain_bootstrap/model_4999.pt
    usd=/home/projects/legacy/robot_v2/usd/v2_plain.usd
  else
    usd=$v14/assets/robot_v2_v14_leg_collision.usda
    options+=(--enable-self-collisions --filter-leg-terrain --measure-leg-clearance)
    actor=$v14/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt
    if [[ "$version" == V15 ]]; then
      actor=$v15/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
      options+=(--joint-drive physx)
    fi
  fi
  sha256sum "$actor" "$usd" "$legacy/runs/v12_distillation_dagger/best.pt" > "$out/${version}_inputs.sha256"
  for terrain in flat stairs_up; do
    terrain_options=()
    warmup=2
    speed=0.4
    if [[ "$terrain" == stairs_up ]]; then
      warmup=0
      speed=0.5
      terrain_options+=(--terrain-level 5 --heading-control)
    fi
    dest=$out/${version}_${terrain}_seed42
    timeout --signal=TERM --kill-after=30s 14m /opt/isaacsim-env/bin/python "$out/evaluate_original.py" \
      --headless --enable_cameras --variant v13 --num-envs 8 --seed 42 \
      --warmup-seconds "$warmup" --seconds 12 --forward-speed "$speed" \
      --scenario "$terrain" --protocol fixed_v2 --effort-guard-fraction 0.99 \
      --actor-checkpoint "$actor" --student-checkpoint "$legacy/runs/v12_distillation_dagger/best.pt" \
      --usd-path "$usd" "${options[@]}" "${terrain_options[@]}" \
      --output "$dest.json" --trace-output "$dest.npz" --trajectory-output "${dest}_trajectory.npz" \
      > "$dest.log" 2>&1
    test -s "$dest.json"
    test -s "$dest.npz"
  done
done
sha256sum "$out"/*.npz "$out"/*.json > "$out/data.sha256"
