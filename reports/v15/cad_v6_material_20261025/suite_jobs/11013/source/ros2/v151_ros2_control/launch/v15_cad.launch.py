"""V15 walking controller for quadruped_20261011; simulator publishes /v15cad sensors."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoints=('actor_checkpoint','student_checkpoint')
    declarations=[DeclareLaunchArgument(k) for k in checkpoints]
    declarations.append(DeclareLaunchArgument('base_com_offset',
        default_value='[0.010427819564938545, 0.015405300073325634, 0.06333552300930023]'))
    parameters={k:LaunchConfiguration(k) for k in checkpoints}
    parameters['base_com_offset']=LaunchConfiguration('base_com_offset')
    return LaunchDescription([*declarations,
        Node(package='v151_ros2_control',executable='v15_walk_node',namespace='v15cad',
             parameters=[parameters],output='screen')])
