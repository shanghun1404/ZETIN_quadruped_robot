#!/bin/bash
set -euo pipefail
root=/home/shanghun1404/v15_cad_20261011
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
usd=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
export PYTHONPATH=$root/config:$root/ros2/v151_ros2_control:$root/ros2/v14_ros2_control:/home/shanghun1404/v15_large_scale/config:/home/shanghun1404/v14_leg_collision/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$ros/rclpy:$usd
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib:$usd/bin
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=97
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export V151_ACTOR=/home/shanghun1404/v15_large_scale/locomotion_jobs/339/guide_v15_physx_pd/2026-09-27_12-47-59_v14_m5700_physx_pd_lane_stairs/model_11699.pt
export V151_STUDENT=/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt
export V151_MODEL=/home/shanghun1404/v151_ros2/model/quadruped.xml
exec /opt/isaacsim-env/bin/python -m pytest -q "$root/config/tests/test_v15_cad_contract.py" \
 "$root/config/tests/test_v15_cad_suite_protocol.py" "$root/config/tests/test_v15_cad_suite_metrics.py" \
 "$root/config/tests/test_v15_cad_model.py" "$root/ros2/v151_ros2_control/test"
