#!/usr/bin/env bash
set -eo pipefail
bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ros_setup=${V15_ROS_SETUP:-/opt/ros/jazzy/setup.bash}
test -f "$ros_setup" || { printf 'ROS2 Jazzy setup missing: %s\n' "$ros_setup" >&2; exit 1; }
source "$ros_setup"
if [ -f "$bundle_root/.venv/bin/activate" ]; then source "$bundle_root/.venv/bin/activate"; fi
python3 "$bundle_root/verify_bundle.py"
python3 "$bundle_root/check_env.py" --output "$bundle_root/environment.json"
export PYTHONPATH="$bundle_root/runtime/config${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import rclpy, numpy, torch, pinocchio, serial; print("Runtime dependencies imported")'
cd "$bundle_root/workspace"
# Running the Python module uses the system-derived virtualenv interpreter for entry points.
python3 -c 'from colcon_core.command import main; raise SystemExit(main())' \
  build --symlink-install --packages-select v14_ros2_control v151_ros2_control v15_hardware_bridge "$@"
