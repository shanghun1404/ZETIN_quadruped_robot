#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=v151-target-screen
#SBATCH --output=/home/shanghun1404/v151_ros2/slurm-target-screen-%j.out
# sbatch compare_targeted.sbatch <targeted-training-job> [original-screen592-directory]
set -euo pipefail
root=/home/shanghun1404/v151_ros2
training_job=${1:?targeted training job required}
[[ "$training_job" =~ ^[0-9]+$ ]]
original=${2:-$root/comparison_jobs/592}
python=/opt/isaacsim-env/bin/python
v14=/home/shanghun1404/v14_leg_collision
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
output=$root/targeted_comparison_jobs/${SLURM_JOB_ID:?Slurm allocation required}
test ! -e "$output"
mkdir -p "$output/source" "$output/analysis"
snapshot=$output/source
sha256sum --check "$original/source.sha256"
sha256sum --check "$original/actors.sha256"
cp -aL "$original/source/." "$snapshot/"
cp "$root/targeted_20261001/analysis/targeted_compare.py" "$output/analysis/"
cp "${BASH_SOURCE[0]}" "$output/launcher.sbatch"
printf '%s\n' "$original" > "$output/frozen_baseline_source.txt"
baseline=$(awk -F '\t' '$1=="baseline339" {print $2}' "$original/actors.tsv")
printf 'baseline339\t%s\n' "$baseline" > "$output/actors.tsv"
for updates in 25 50 100; do
    printf -v padded '%04d' "$updates"
    mapfile -t paths < <(find "$root/targeted_training_jobs/$training_job" -name "pilot_update_$padded.pt" -type f)
    [[ ${#paths[@]} == 1 ]]
    printf 'candidate%s\t%s\n' "$updates" "${paths[0]}" >> "$output/actors.tsv"
done
while IFS=$'\t' read -r label actor; do sha256sum "$actor"; done < "$output/actors.tsv" > "$output/actors.sha256"
$python - "$original" "$output" <<'PY'
import json,pathlib,shutil,sys
old,new=map(pathlib.Path,sys.argv[1:])
for terrain in ('flat','rough'):
    name=f'baseline339_{terrain}_seed42';run=old/name
    result=json.loads((run/'result.json').read_text())
    assert result['completed'] and not result['failure'] and result['seed']==42 and result['scenario']==terrain
    assert int((run/'exit_code.txt').read_text())==0 and (run/'result.npz').is_file()
    shutil.copytree(run,new/name)
PY
find "$snapshot" "$output/analysis" -type f -not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' -print0 | sort -z | xargs -0 sha256sum > "$output/source.sha256"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$snapshot/config:$snapshot/ros2/v151_ros2_control:$v14/ros2/v14_ros2_control:/home/shanghun1404/v15_large_scale/config:$v14/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
cd "$snapshot"
$python -m pytest -q "$snapshot/ros2/v151_ros2_control/test/test_motion.py" "$snapshot/ros2/v151_ros2_control/test/test_standing.py" "$snapshot/reports/v151/analysis/test_compare_ros2.py" > "$output/tests.log" 2>&1
controller_pid=
stop_controller() {
    if [[ -n "$controller_pid" ]]; then
        kill "$controller_pid" 2>/dev/null || true
        sleep .5
        kill -KILL "$controller_pid" 2>/dev/null || true
        wait "$controller_pid" 2>/dev/null || true
        controller_pid=
    fi
}
trap stop_controller EXIT
run_index=0
run_case() {
    local label=$1 terrain=$2 actor run namespace status
    actor=$(awk -F '\t' -v label="$label" '$1==label {print $2}' "$output/actors.tsv")
    run_index=$((run_index+1))
    run=$output/${label}_${terrain}_seed42
    mkdir "$run"
    export ROS_DOMAIN_ID=$(((SLURM_JOB_ID+run_index) % 80 + 10))
    namespace=/v151_target_${SLURM_JOB_ID}_${run_index}
    printf '%s\n' "$actor" > "$run/actor.txt"
    printf '%s\n' "$ROS_DOMAIN_ID" > "$run/ros_domain.txt"
    $python -m v151_ros2_control.controller_node --ros-args -r __ns:="$namespace" \
        -p actor_checkpoint:="$actor" \
        -p student_checkpoint:=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt \
        -p model_path:="$snapshot/model/quadruped.xml" -p initial_mode:=WALK -p demo:=false -p speed:=0.4 \
        > "$run/controller.log" 2>&1 &
    controller_pid=$!
    set +e
    timeout --signal=TERM --kill-after=15s 12m "$python" "$snapshot/config/evaluate_v151_ros2.py" \
        --headless --enable_cameras --usd-path "$v14/assets/robot_v2_v14_leg_collision.usda" \
        --output "$run/result.json" --seconds 14 --seed 42 --scenario "$terrain" --namespace "$namespace" \
        > "$run/evaluator.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$run/exit_code.txt"
    stop_controller
}
for label in candidate25 candidate50 candidate100; do
    run_case "$label" flat
    set +e
    "$python" "$output/analysis/targeted_compare.py" "$output" --frozen-source "$snapshot" --flat "$label"
    decision=$?
    set -e
    if [[ "$decision" == 0 ]]; then
        run_case "$label" rough
    elif [[ "$decision" == 10 ]]; then
        printf 'Rejected %s on flat; rough skipped without adoption.\n' "$label"
    else
        exit "$decision"
    fi
done
"$python" "$output/analysis/targeted_compare.py" "$output" --frozen-source "$snapshot"
