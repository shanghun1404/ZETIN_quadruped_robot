"""Jetson launch: serial feedback, fresh sensor snapshots, CAD controller."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    required=('actor_checkpoint','student_checkpoint','model_path','hardware_config')
    declarations=[DeclareLaunchArgument(name) for name in required]
    declarations.extend([DeclareLaunchArgument('namespace',default_value='v15real'),
                         DeclareLaunchArgument('enable_motors',default_value='false')])
    namespace=LaunchConfiguration('namespace')
    shared=dict(use_sim_time=False)
    return LaunchDescription([*declarations,
        Node(package='v15_hardware_bridge',executable='usb_gateway',namespace=namespace,output='screen',parameters=[dict(**shared,
            hardware_config=LaunchConfiguration('hardware_config'),
            enable_motors=ParameterValue(LaunchConfiguration('enable_motors'),value_type=bool))]),
        Node(package='v15_hardware_bridge',executable='sensor_sync',namespace=namespace,output='screen',parameters=[shared]),
        Node(package='v15_hardware_bridge',executable='cad_controller',namespace=namespace,output='screen',parameters=[dict(**shared,
            actor_checkpoint=LaunchConfiguration('actor_checkpoint'),student_checkpoint=LaunchConfiguration('student_checkpoint'),
            model_path=LaunchConfiguration('model_path'),demo=False,initial_mode='SIT',speed=0.,
            stand_forward_shift_m=0.,crouch_forward_shift_m=0.,crouch_shift_after_walk_only=True)])])
