from setuptools import setup
from glob import glob

setup(name='v15_hardware_bridge',version='0.1.0',packages=['v15_hardware_bridge'],
      data_files=[('share/ament_index/resource_index/packages',['resource/v15_hardware_bridge']),
                  ('share/v15_hardware_bridge',['package.xml']),
                  ('share/v15_hardware_bridge/launch',glob('launch/*.py')),
                  ('share/v15_hardware_bridge/config',glob('config/*.json'))],
      install_requires=['setuptools','pyserial'],zip_safe=True,
      entry_points={'console_scripts':['usb_gateway = v15_hardware_bridge.usb_gateway:main',
          'sensor_sync = v15_hardware_bridge.sensor_sync:main',
          'cad_controller = v15_hardware_bridge.cad_controller:main']})
