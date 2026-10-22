#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=v15-torque-reward-pilot
#SBATCH --output=/home/shanghun1404/v15_large_scale/slurm-%j.out
# PREPARED ONLY: parent reviews/submits. Defaults match339 scale for250 updates.
set -euo pipefail
num_envs=${1:-512}
iterations=${2:-250}
[[ "$num_envs" -eq 512 ]] || exit 2
[[ "$iterations" -eq 250 ]] || exit 2
v15=/home/shanghun1404/v15_large_scale
v14=/home/shanghun1404/v14_leg_collision
audit_config=$v15/config/experiments/v16_20260929
audit_output=$v15/results/v16_20260929/v15_torque_reward_finetune_${SLURM_JOB_ID}
actor=$v15/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH=$v15/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$v14/ros2/v14_ros2_control
mkdir -p "$audit_output"
cp "$audit_config/v15_torque_reward_finetune.py" "$audit_config/v15_torque_reward_finetune.sbatch" "$audit_output/"
sha256sum "$actor" "$v15/config/train_v15_locomotion.py" "$audit_output"/v15_torque_* \
  "$v15/config/v15_env_cfg.py" "$v15/config/v15_math.py" \
  "$v14/config/v14_leg_env_cfg.py" /home/shanghun1404/v13_torque/config/v13_torque_actions.py \
  /home/shanghun1404/v13_torque/config/v13_torque_mdp.py \
  /home/projects/legacy/quadruped_training_smooth_v11/v8_env_cfg.py \
  /opt/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py \
  /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  "$v14/assets/robot_v2_v14_leg_collision.usda" > "$audit_output/source_sha256.txt"
cd "$v15/config"
/opt/isaacsim-env/bin/python "$audit_output/v15_torque_reward_finetune.py" \
  --headless --enable_cameras --resume --num_envs "$num_envs" --max_iterations "$iterations" \
  --seed 47 --max-delay-steps 3 --output-root "$audit_output" \
  --actor-checkpoint "$actor" \
  --student-checkpoint /home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
  --usd-path "$v14/assets/robot_v2_v14_leg_collision.usda" > "$audit_output/pilot.log" 2>&1
run_dir=$(find "$audit_output/guide_v15_torque_reward_finetune" -mindepth 1 -maxdepth 1 -type d -print -quit)
test -s "$run_dir/run_completed.json"
test -s "$run_dir/finetune_actual_contract.json"
sha256sum "$run_dir/run_completed.json" "$run_dir/finetune_actual_contract.json" "$run_dir/v15_manifest.json" "$run_dir"/model_*.pt > "$audit_output/data_sha256.txt"

sha256sum --quiet -c "$audit_output/source_sha256.txt"
