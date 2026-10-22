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
                         DeclareLaunchArgument('enable_motors',default_value='false'),
                         DeclareLaunchArgument('device',default_value='cuda'),
                         DeclareLaunchArgument('wbc_backend',default_value='cpp'),
                         DeclareLaunchArgument('initial_mode',default_value='SIT'),
                         DeclareLaunchArgument('sit_style',default_value='crouch'),
                         DeclareLaunchArgument('height_drop',default_value='0.10'),
                         DeclareLaunchArgument('posture_transition_s',default_value='3.0')])
    namespace=LaunchConfiguration('namespace')
    shared=dict(use_sim_time=False)
    return LaunchDescription([*declarations,
        Node(package='v15_hardware_bridge',executable='usb_gateway',namespace=namespace,output='screen',parameters=[dict(**shared,
            hardware_config=LaunchConfiguration('hardware_config'),
            enable_motors=ParameterValue(LaunchConfiguration('enable_motors'),value_type=bool))]),
        Node(package='v15_hardware_bridge',executable='sensor_sync',namespace=namespace,output='screen',parameters=[shared]),
        Node(package='v15_hardware_bridge',executable='joint_command_node',namespace=namespace,output='screen',parameters=[shared]),
        Node(package='v15_hardware_bridge',executable='cad_controller',namespace=namespace,output='screen',
            remappings=[('joint_states','policy_joint_states')],parameters=[dict(**shared,
            planner_only=True,device=LaunchConfiguration('device'),wbc_backend=LaunchConfiguration('wbc_backend'),
            actor_checkpoint=LaunchConfiguration('actor_checkpoint'),student_checkpoint=LaunchConfiguration('student_checkpoint'),
            model_path=LaunchConfiguration('model_path'),demo=False,initial_mode=LaunchConfiguration('initial_mode'),speed=0.,
            sit_style=LaunchConfiguration('sit_style'),
            height_drop=ParameterValue(LaunchConfiguration('height_drop'),value_type=float),
            posture_transition_s=ParameterValue(LaunchConfiguration('posture_transition_s'),value_type=float),
            stand_forward_shift_m=0.,crouch_forward_shift_m=0.,crouch_shift_after_walk_only=True)])])
