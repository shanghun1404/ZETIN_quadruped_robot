#!/bin/bash
# HPC ws1: V14 ROS2 package tests with Isaac Sim's bundled ROS2 Jazzy rclpy (no system ROS2).
set -euo pipefail
root=/home/shanghun1404/v14_leg_collision
ros=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy
export PYTHONPATH=$root/config:/home/shanghun1404/v13_torque/config:/home/projects/legacy/quadruped_training_smooth_v11:$root/ros2/v14_ros2_control:$ros/rclpy
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$ros/lib
export ROS_DOMAIN_ID=$(( ${SLURM_JOB_ID:-$$} % 90 + 10 ))
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1  # the bundled launch_testing plugin needs lark
cd $root/ros2/v14_ros2_control
/opt/isaacsim-env/bin/python -m pytest -q -p no:cacheprovider test "$@"
