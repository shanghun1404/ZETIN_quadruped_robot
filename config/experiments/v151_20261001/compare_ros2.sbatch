#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=v151-compare
#SBATCH --output=/home/shanghun1404/v151_ros2/slurm-compare-%j.out
# Screen: sbatch compare_ros2.sbatch screen 589
# Resume: sbatch compare_ros2.sbatch resume /home/shanghun1404/v151_ros2/comparison_jobs/592
# Holdout: sbatch compare_ros2.sbatch holdout /home/shanghun1404/v151_ros2/comparison_jobs/<screen-job>
set -euo pipefail
root=/home/shanghun1404/v151_ros2
v14=/home/shanghun1404/v14_leg_collision
python=/opt/isaacsim-env/bin/python
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
mode=${1:-screen}
analysis_mode=$mode
output=$root/comparison_jobs/${SLURM_JOB_ID:?Slurm allocation required}
test ! -e "$output"
mkdir -p "$output/source"
snapshot=$output/source
if [[ "$mode" == screen ]]; then
    training_job=${2:-589}
    [[ "$training_job" =~ ^[0-9]+$ ]]
    mapfile -t candidate250 < <(find "$root/training_jobs/$training_job" -name pilot_update_0250.pt -type f)
    mapfile -t candidate500 < <(find "$root/training_jobs/$training_job" -name pilot_update_0500.pt -type f)
    [[ ${#candidate250[@]} == 1 && ${#candidate500[@]} == 1 ]]
    baseline=/home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
    printf 'baseline339\t%s\ncandidate250\t%s\ncandidate500\t%s\n' "$baseline" "${candidate250[0]}" "${candidate500[0]}" > "$output/actors.tsv"
    # Freeze current parent integration only when the batch starts, after smoke validation.
    cp -aL "$root/config" "$root/ros2" "$root/model" "$snapshot/"
    mkdir -p "$snapshot/reports/v16" "$snapshot/reports/v151"
    cp -aL "$root/reports/v16/physical_model" "$snapshot/reports/v16/"
    cp -aL "$root/reports/v151/analysis" "$snapshot/reports/v151/"
    labels=(baseline339 candidate250 candidate500)
    seeds=(42)
elif [[ "$mode" == resume ]]; then
    screening=${2:?screening output directory required}
    # Preserve the old job. A resume owns a new directory and the old frozen source.
    sha256sum --check "$screening/source.sha256"
    sha256sum --check "$screening/actors.sha256"
    cp -aL "$screening/source/." "$snapshot/"
    cp "$screening/actors.tsv" "$output/actors.tsv"
    printf '%s\n' "$screening" > "$output/resumed_from.txt"
    $python - "$screening" "$output" <<'PY'
import json, pathlib, shutil, sys
old, new = map(pathlib.Path, sys.argv[1:])
reused = []
for label in ('baseline339', 'candidate250', 'candidate500'):
    for terrain in ('flat', 'rough'):
        name = f'{label}_{terrain}_seed42'
        run = old / name
        try:
            result = json.loads((run / 'result.json').read_text())
            complete = (int((run / 'exit_code.txt').read_text()) == 0
                        and result.get('completed') is True and not result.get('failure')
                        and (run / 'result.npz').is_file() and (run / 'result.status.json').is_file())
        except (OSError, ValueError, TypeError):
            complete = False
        if complete:
            shutil.copytree(run, new / name)
            reused.append(name)
(new / 'reused_runs.json').write_text(json.dumps(reused, indent=2) + '\n')
print('Reused completed runs:', reused, flush=True)
PY
    labels=(baseline339 candidate250 candidate500)
    seeds=(42)
    analysis_mode=screen
elif [[ "$mode" == holdout ]]; then
    screening=${2:?screening output directory required}
    selected=$($python - "$screening/comparison_summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['mode']=='screen' and d['all_gates_pass'] and d['selected'] in ('candidate250','candidate500')
print(d['selected'])
PY
)
    # Heldout seeds reuse exactly the sources evaluated at seed42.
    cp -aL "$screening/source/." "$snapshot/"
    cp "$screening/actors.tsv" "$output/actors.tsv"
    sha256sum --check "$screening/actors.sha256"
    printf '%s\n' "$screening" > "$output/screening_source.txt"
    labels=(baseline339 "$selected")
    seeds=(43 44)
else
    echo 'mode must be screen, resume or holdout' >&2
    exit 2
fi
cp "${BASH_SOURCE[0]}" "$output/launcher.sbatch"
find "$snapshot" -type f -not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' -print0 | sort -z | xargs -0 sha256sum > "$output/source.sha256"
while IFS=$'\t' read -r label actor; do
    test -s "$actor"
    sha256sum "$actor"
done < "$output/actors.tsv" > "$output/actors.sha256"
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
for seed in "${seeds[@]}"; do
    for label in "${labels[@]}"; do
        actor=$(awk -F '\t' -v label="$label" '$1==label {print $2}' "$output/actors.tsv")
        for terrain in flat rough; do
            run_index=$((run_index+1))
            run=$output/${label}_${terrain}_seed${seed}
            if [[ "$mode" == resume && -d "$run" ]]; then
                printf 'Skipping preserved completed run: %s\n' "$run"
                continue
            fi
            mkdir "$run"
            export ROS_DOMAIN_ID=$(((SLURM_JOB_ID+run_index) % 80 + 10))
            namespace=/v151_compare_${SLURM_JOB_ID}_${run_index}
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
                --output "$run/result.json" --seconds 14 --seed "$seed" --scenario "$terrain" --namespace "$namespace" \
                > "$run/evaluator.log" 2>&1
            status=$?
            set -e
            printf '%s\n' "$status" > "$run/exit_code.txt"
            stop_controller
        done
    done
done
analysis_args=(--mode "$analysis_mode")
if [[ "$mode" == holdout ]]; then analysis_args+=(--selected "$selected"); fi
$python "$snapshot/reports/v151/analysis/compare_ros2.py" "$output" "${analysis_args[@]}"
