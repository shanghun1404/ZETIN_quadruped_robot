#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=v15-videos
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# Record V14/V15 comparison videos (flat, stairs) and V15-R recoveries into results/videos/<label>/.
#   sbatch launch_v15_videos.sbatch <new label>
set -euo pipefail
label=${1:?new recording label required}
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
out=$v15/results/videos/$label
test ! -e "$out"
mkdir -p "$out"
locomotion=$(cat $v15/locomotion_jobs/339/checkpoint.txt)
recovery=$(cat $v15/recovery_jobs/416/checkpoint.txt)
m5700=$v14/pilot_jobs/110/guide_v14_interleg_collision/2026-09-24_22-59-31_v11_actor_v12_depth_guard_099/model_5700.pt
student=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
evaluate=$v15/config/launch_v15_eval.sbatch

# PhysX joint-drive PD (the ROS2 plant): V15 main and the V14 5700 starting point.
bash $evaluate "$locomotion" "video_${label}_v15" flat 42 --video-dir "$out/v15_flat" --video-env-ids 0 1
bash $evaluate "$locomotion" "video_${label}_v15_heading" stairs_up 42 --heading-control \
    --video-dir "$out/v15_stairs_up" --video-env-ids 0 1
bash $evaluate "$m5700" "video_${label}_v14_physx" flat 42 --video-dir "$out/v14_physx_flat" --video-env-ids 0 1

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
cd $v15/config
# V14 5700 on its training actuator (explicit DCMotor), same flat protocol.
/opt/isaacsim-env/bin/python evaluate_v13_torque.py \
  --headless --enable_cameras --variant v13 --num-envs 32 --seed 42 \
  --warmup-seconds 2 --seconds 12 --forward-speed 0.4 --scenario flat --protocol fixed_v2 \
  --effort-guard-fraction 0.99 --enable-self-collisions --filter-leg-terrain --measure-leg-clearance \
  --actor-checkpoint "$m5700" --student-checkpoint "$student" --usd-path $v14/assets/robot_v2_v14_leg_collision.usda \
  --output "$out/v14_dcmotor_flat.json" --video-dir "$out/v14_dcmotor_flat" --video-env-ids 0 1
# V15-R from six random falls.
/opt/isaacsim-env/bin/python evaluate_v15_recovery.py --headless --enable_cameras --checkpoint "$recovery" \
  --usd-path $v15/assets/robot_v2_v15_recovery.usda --start fallen --seed 42 \
  --output "$out/v15_recovery_fallen.json" --video-dir "$out/v15_recovery" --video-env-ids 0 1 2 3 4 5
find "$out" -name '*.mp4' -printf '%p %s\n'
