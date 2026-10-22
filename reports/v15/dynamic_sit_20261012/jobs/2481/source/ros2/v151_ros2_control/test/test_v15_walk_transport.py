"""Real DDS WALK transport compared with the released policy and motor law."""
import array
from copy import deepcopy
import json
import os
import time
from unittest.mock import patch

import numpy as np
import pytest
import torch


@pytest.mark.skipif(not os.environ.get('V151_ACTOR'), reason='requires explicit checkpoint/runtime paths')
def test_walk_matches_released_policy_and_replays_without_advancing_control():
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState, Image
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from v14_ros2_control import contract
    from v14_ros2_control.policy import V14DepthPolicy
    from v14_ros2_control.joint_command import JointCommandFilter
    from v151_ros2_control.motion import CANONICAL, hybrid_motor_command
    from v151_ros2_control.v15_walk_node import V15WalkNode

    offset = np.array([.01042782, .0154053, .06333552])
    origin_velocity = np.array([.04, .26, -.03])
    omega = np.array([.3, -.2, .7])
    rclpy.init()
    ctl = tester = executor = None
    try:
        ctl = V15WalkNode(namespace='/cad_walk_test', parameter_overrides=[
            Parameter('actor_checkpoint', value=os.environ['V151_ACTOR']),
            Parameter('student_checkpoint', value=os.environ['V151_STUDENT']),
            Parameter('base_com_offset', value=offset.tolist()),
        ])
        tester = rclpy.create_node('walk_test_sensors', namespace='/cad_walk_test')
        executor = SingleThreadedExecutor()
        executor.add_node(ctl); executor.add_node(tester)
        qos = contract.lockstep_qos(); received = []; statuses = []
        pubs = [tester.create_publisher(typ, topic, qos) for typ, topic in (
            (JointState, 'joint_states'), (Odometry, 'odom'), (Image, 'depth/image'),
            (JointState, 'foot_contacts'))]
        cmd_pub = tester.create_publisher(Twist, 'cmd_vel', 10)
        tester.create_subscription(JointState, 'joint_command', received.append, qos)
        tester.create_subscription(String, 'status', statuses.append, qos)
        def spin_until(predicate):
            deadline = time.monotonic() + 10.
            while not predicate():
                executor.spin_once(timeout_sec=.01)
                assert time.monotonic() < deadline
        spin_until(lambda: all(p.get_subscription_count() for p in pubs) and
                   cmd_pub.get_subscription_count() and ctl.pub.get_subscription_count() and
                   ctl.status_pub.get_subscription_count())
        offline = V14DepthPolicy(os.environ['V151_ACTOR'], os.environ['V151_STUDENT'])
        filt = JointCommandFilter()
        to_canonical = [contract.JOINT_NAMES.index(n) for n in CANONICAL]
        to_policy = [CANONICAL.index(n) for n in contract.JOINT_NAMES]
        q = np.array([contract.joint_default(n) for n in contract.JOINT_NAMES]) + np.linspace(-.03, .03, 12)
        dq = np.linspace(-.1, .15, 12)
        state = JointState(name=list(reversed(contract.JOINT_NAMES)), position=q[::-1].tolist(), velocity=dq[::-1].tolist())
        odom = Odometry(); odom.pose.pose.orientation.w = 1.
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = origin_velocity.tolist()
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = omega.tolist()
        depth = np.linspace(.2, 4., 84*84, dtype=np.float32).reshape(84, 84)
        image = Image(height=84, width=84, encoding='32FC1', is_bigendian=0, step=336,
                      data=array.array('B', depth.astype('<f4').tobytes()))
        contacts = JointState(name=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'], effort=[50., 0., 55., 60.])
        filt.reset(torch.tensor(q, dtype=torch.float32))
        command = [0., .4, 0.]
        for step in range(5):
            if step == 4:
                command = [.02, .31, -.08]
                twist = Twist(); twist.linear.x, twist.linear.y, twist.angular.z = command
                cmd_pub.publish(twist)
                spin_until(lambda: ctl.command == command)
            sec, ns = contract.step_stamp(step); stamp = sec*1_000_000_000 + ns
            state.header.stamp.sec, state.header.stamp.nanosec = sec, ns
            before = len(received); before_status = len(statuses)
            if contract.is_policy_tick(stamp):
                for msg in (odom, image, contacts): msg.header.stamp = state.header.stamp
                for i in (0, 1, 3, 0): pubs[i].publish((state, odom, image, contacts)[i])
                for _ in range(16): executor.spin_once(timeout_sec=.005)
                assert len(received) == before and len(statuses) == before_status
                pubs[2].publish(image)
                _, target = offline.step(joint_pos=q, joint_vel=dq,
                    lin_vel_b=origin_velocity + np.cross(omega, offset), ang_vel_b=omega,
                    quat_wxyz=[1., 0., 0., 0.], command=command, depth_m=depth)
                filt.set_target(target)
            else:
                pubs[0].publish(state)
            filtered = filt.step(torch.tensor(q, dtype=torch.float32), torch.tensor(dq, dtype=torch.float32)).numpy()
            expected = hybrid_motor_command(1., np.zeros(12), filtered[to_canonical], q[to_canonical], dq[to_canonical])
            spin_until(lambda: len(received) > before and len(statuses) > before_status)
            np.testing.assert_allclose(received[-1].position, expected['position'][to_policy], atol=1e-6)
            np.testing.assert_array_equal(received[-1].effort, np.zeros(12))
            status = json.loads(statuses[-1].data)
            assert status['stamp_ns'] == stamp and status['mode'] == 'WALK'
            assert status['walk_weight'] == status['height_fraction'] == 1.
            assert status['command_schema'] == 2 and status['support_feet'] == 3
            assert status['stiffness_nm_rad'] == 300. and status['damping_nms_rad'] == 8.
            np.testing.assert_allclose(status['requested_torque'], expected['requested_torque'], atol=1e-6)
            np.testing.assert_allclose(status['bounded_torque'], expected['bounded_torque'], atol=1e-6)
            saved = deepcopy(received[-1]); saved_status = statuses[-1].data
            count, count_status = len(received), len(statuses)
            with patch.object(ctl.policy, 'step', side_effect=AssertionError('policy rerun')), \
                 patch.object(ctl.filter, 'step', side_effect=AssertionError('filter rerun')):
                pubs[0].publish(state)
                spin_until(lambda: len(received) > count and len(statuses) > count_status)
            assert received[-1] == saved and statuses[-1].data == saved_status
        count, count_status = len(received), len(statuses)
        state.header.stamp.nanosec = 0; pubs[0].publish(state)
        for _ in range(16): executor.spin_once(timeout_sec=.005)
        assert (len(received), len(statuses)) == (count, count_status)
        assert ctl.policy_steps == 2
    finally:
        if executor is not None: executor.shutdown()
        if ctl is not None: ctl.destroy_node()
        if tester is not None: tester.destroy_node()
        rclpy.shutdown()
