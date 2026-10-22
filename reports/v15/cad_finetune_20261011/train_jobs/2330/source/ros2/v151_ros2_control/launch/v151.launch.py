from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    keys=('actor_checkpoint','student_checkpoint','model_path')
    return LaunchDescription([*(DeclareLaunchArgument(k) for k in keys),
        Node(package='v151_ros2_control',executable='controller_node',namespace='v151',
             parameters=[{k:LaunchConfiguration(k) for k in keys}],output='screen')])
