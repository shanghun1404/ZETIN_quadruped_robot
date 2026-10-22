"""Actual DDS integration with a synthetic sensor publisher (no simulated physics)."""
import json
import os
import time
import array
import numpy as np
import pytest

@pytest.mark.skipif(not os.environ.get('V151_ACTOR'), reason='requires explicit checkpoint/runtime paths')
def test_sensor_stamp_drives_one_command_and_exposes_wbc_diagnostics():
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState, Image
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from v14_ros2_control import contract
    from v151_ros2_control.controller_node import ControllerNode
    rclpy.init(args=['--ros-args','-p','actor_checkpoint:='+os.environ['V151_ACTOR'],
        '-p','student_checkpoint:='+os.environ['V151_STUDENT'],'-p','model_path:='+os.environ['V151_MODEL']])
    ctl=ControllerNode();tester=rclpy.create_node('v151_test_sensors')
    executor=SingleThreadedExecutor();executor.add_node(ctl);executor.add_node(tester)
    qos=contract.lockstep_qos();received=[];statuses=[]
    pubs=[tester.create_publisher(typ,topic,qos) for typ,topic in
        [(JointState,'joint_states'),(Odometry,'odom'),(Image,'depth/image'),(JointState,'foot_contacts')]]
    tester.create_subscription(JointState,'joint_command',received.append,qos)
    tester.create_subscription(String,'status',statuses.append,qos)
    try:
        deadline=time.monotonic()+10
        while any(p.get_subscription_count()<1 for p in pubs):
            executor.spin_once(timeout_sec=.01)
            assert time.monotonic()<deadline
        state=JointState();state.name=list(contract.JOINT_NAMES)
        state.position=[contract.joint_default(n) for n in contract.JOINT_NAMES];state.velocity=[0.]*12
        odom=Odometry();odom.pose.pose.position.z=.464;odom.pose.pose.orientation.w=1.
        image=Image();image.height=84;image.width=84;image.encoding='32FC1';image.step=84*4
        image.data=array.array('B',np.full((84,84),2.,dtype='<f4').tobytes())
        contacts=JointState();contacts.name=['FL_foot','FR_foot','RL_foot','RR_foot'];contacts.effort=[111.]*4
        # Deliberately out-of-order: state arrives before image.
        for pub,msg in zip(pubs,[state,odom,image,contacts]): pub.publish(msg)
        while not received or not statuses:
            executor.spin_once(timeout_sec=.01)
            assert time.monotonic()<deadline
        assert len(received)==1
        assert tuple(received[0].name)==contract.JOINT_NAMES
        assert np.isfinite(received[0].position).all()
        status=json.loads(statuses[0].data)
        assert status['support_feet']==4
        assert status['wbc']['finite']
        assert status['wbc']['feasible']
        # Duplicate state cannot emit an extra command.
        pubs[0].publish(state)
        for _ in range(10): executor.spin_once(timeout_sec=.005)
        assert len(received)==1
    finally:
        executor.shutdown();ctl.destroy_node();tester.destroy_node();rclpy.shutdown()
