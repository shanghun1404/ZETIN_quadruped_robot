from glob import glob

from setuptools import setup

package_name = "v14_ros2_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="V14 quadruped policy and joint command nodes driving Isaac Sim over ROS2.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = v14_ros2_control.policy_node:main",
            "joint_command_node = v14_ros2_control.joint_command_node:main",
        ],
    },
)
