from setuptools import setup
from glob import glob
setup(name='v151_ros2_control',version='0.1.0',packages=['v151_ros2_control'],
 data_files=[('share/ament_index/resource_index/packages',['resource/v151_ros2_control']),
 ('share/v151_ros2_control',['package.xml']),('share/v151_ros2_control/launch',glob('launch/*.py'))],
 install_requires=['setuptools','numpy'],zip_safe=True,
 entry_points={'console_scripts':['controller_node = v151_ros2_control.controller_node:main']})
