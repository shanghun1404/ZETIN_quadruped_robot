"""Real ROS2 boundary: aligned measured inputs, no fabricated or skipped ticks."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
if importlib.util.find_spec('rclpy') is None:
    pytest.skip('native ROS2 Jazzy test',allow_module_level=True)
sys.path.insert(0,str(Path(__file__).parents[1]))
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState,Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from v14_ros2_control import contract


def test_measured_sensor_snapshot_and_missing_tick_stops_output():
    assert (Path(__file__).parents[1]/'v15_hardware_bridge/sensor_sync.py').exists(),'sensor adapter is missing'
    from v15_hardware_bridge.sensor_sync import SensorSync
    rclpy.init();sync=SensorSync();probe=rclpy.create_node('sensor_test',namespace='/v15real')
    types=dict(joints=JointState,odom=Odometry,depth=Image,contacts=JointState)
    topics=dict(joints='raw/joint_states',odom='raw/odom',depth='raw/depth/image',contacts='raw/foot_contacts')
    pubs={n:probe.create_publisher(types[n],topics[n],10) for n in types}
    outputs={n:[] for n in types};faults=[]
    for name,topic in dict(joints='joint_states',odom='odom',depth='depth/image',contacts='foot_contacts').items():
        probe.create_subscription(types[name],topic,outputs[name].append,contract.lockstep_qos())
    probe.create_subscription(String,'hardware_fault',faults.append,10)
    executor=SingleThreadedExecutor();executor.add_node(sync);executor.add_node(probe)
    try:
        deadline=time.monotonic()+2.
        while any(p.get_subscription_count()==0 for p in pubs.values()) and time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.001)
        stamp=probe.get_clock().now().to_msg()
        odom=Odometry();odom.header.stamp=stamp;odom.header.frame_id='world';odom.child_frame_id='base';odom.pose.pose.orientation.w=1.
        depth=Image();depth.header.stamp=stamp;depth.height=depth.width=84;depth.encoding='32FC1';depth.step=336;depth.data=bytes(84*84*4)
        contacts=JointState();contacts.header.stamp=stamp;contacts.name=['FL_foot','FR_foot','RL_foot','RR_foot'];contacts.effort=[30.]*4
        for n,m in [('odom',odom),('depth',depth),('contacts',contacts)]:pubs[n].publish(m)
        for _ in range(12):executor.spin_once(timeout_sec=.001)
        q=JointState();q.header.stamp.nanosec=20_000_000;q.name=list(contract.JOINT_NAMES);q.position=[0.]*12;q.velocity=[0.]*12;q.effort=[0.]*12
        pubs['joints'].publish(q)
        deadline=time.monotonic()+.5
        while not all(outputs.values()) and time.monotonic()<deadline:executor.spin_once(timeout_sec=.001)
        assert all(outputs.values())
        assert {contract.stamp_ns(outputs[n][0].header.stamp) for n in outputs}=={0}
        assert outputs['contacts'][0].effort==pytest.approx([30.]*4)
        q.header.stamp.nanosec=30_000_000;pubs['joints'].publish(q)
        deadline=time.monotonic()+.5
        while not faults and time.monotonic()<deadline:executor.spin_once(timeout_sec=.001)
        assert faults and 'missing' in faults[0].data
        assert len(outputs['joints'])==1
    finally:
        sync.destroy_node();probe.destroy_node();executor.shutdown();rclpy.shutdown()
