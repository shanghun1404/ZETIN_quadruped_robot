#!/usr/bin/env bash
set -eo pipefail
bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ros_setup=${V15_ROS_SETUP:-/opt/ros/jazzy/setup.bash}
test -f "$ros_setup" || { printf 'ROS2 Jazzy setup missing: %s\n' "$ros_setup" >&2; exit 1; }
test -f "$bundle_root/workspace/install/setup.bash" || { printf 'Run build.sh first.\n' >&2; exit 1; }
source "$ros_setup"
if [ -f "$bundle_root/.venv/bin/activate" ]; then source "$bundle_root/.venv/bin/activate"; fi
source "$bundle_root/workspace/install/setup.bash"
python3 "$bundle_root/verify_bundle.py"
export PYTHONPATH="$bundle_root/runtime/config${PYTHONPATH:+:$PYTHONPATH}"
hardware_config=${V15_HARDWARE_CONFIG:-$bundle_root/workspace/src/v15_hardware_bridge/config/hardware.json}
hardware_config=$(realpath -- "$hardware_config")
enable_motors=${V15_ENABLE_MOTORS:-false}
case "$enable_motors" in true|false) ;; *) printf 'V15_ENABLE_MOTORS must be true or false.\n' >&2; exit 1;; esac
test -f "$hardware_config" || { printf 'Hardware configuration missing: %s\n' "$hardware_config" >&2; exit 1; }
exec ros2 launch v15_hardware_bridge hardware.launch.py \
  "actor_checkpoint:=$bundle_root/checkpoints/model_15699.pt" \
  "student_checkpoint:=$bundle_root/checkpoints/v12_student_best.pt" \
  "model_path:=$bundle_root/model/quadruped_cad.xml" \
  "hardware_config:=$hardware_config" "enable_motors:=$enable_motors" "$@"
