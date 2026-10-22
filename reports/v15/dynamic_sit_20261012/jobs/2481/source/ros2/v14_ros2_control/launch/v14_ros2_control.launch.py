"""Start the V14 policy and joint command nodes; Isaac Sim is launched separately."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    # The nodes import the V12/V13 modules (v13_depth_policy, v13_torque_mdp, ...) from config/.
    config_dir = LaunchConfiguration("config_dir")
    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="/v14"),
        DeclareLaunchArgument("config_dir", description="robot_dog config/ directory"),
        DeclareLaunchArgument("actor_checkpoint"),
        DeclareLaunchArgument("student_checkpoint"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("command", default_value="[0.0, 0.4, 0.0]"),
        DeclareLaunchArgument("guard_fraction", default_value="0.99"),
        SetEnvironmentVariable("PYTHONPATH", [config_dir, os.pathsep, os.environ.get("PYTHONPATH", "")]),
        Node(
            package="v14_ros2_control", executable="joint_command_node", namespace=namespace,
            parameters=[{"guard_fraction": LaunchConfiguration("guard_fraction")}],
            output="screen",
        ),
        Node(
            package="v14_ros2_control", executable="policy_node", namespace=namespace,
            parameters=[{
                "actor_checkpoint": LaunchConfiguration("actor_checkpoint"),
                "student_checkpoint": LaunchConfiguration("student_checkpoint"),
                "device": LaunchConfiguration("device"),
                "command": LaunchConfiguration("command"),
            }],
            output="screen",
        ),
    ])
