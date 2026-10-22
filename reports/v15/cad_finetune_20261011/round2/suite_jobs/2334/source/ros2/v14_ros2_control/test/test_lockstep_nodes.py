"""Both ROS2 nodes against a fake lockstep simulator over real DDS topics (no Isaac Sim)."""

import numpy as np
import pytest
import torch

rclpy = pytest.importorskip("rclpy")
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import Image, JointState  # noqa: E402

from v14_ros2_control import contract  # noqa: E402
from v14_ros2_control.joint_command import JointCommandFilter  # noqa: E402
from v14_ros2_control.joint_command_node import V14JointCommandNode  # noqa: E402
from v14_ros2_control.policy import V14DepthPolicy  # noqa: E402
from v14_ros2_control.policy_node import V14PolicyNode  # noqa: E402

from test_policy import rsl_actor_checkpoint, student_checkpoint  # noqa: E402


def test_nodes_answer_every_physics_step_with_the_offline_control_law(tmp_path):
    rsl_actor_checkpoint(tmp_path / "actor.pt")
    student_checkpoint(tmp_path / "student.pt")
    rclpy.init()
    try:
        namespace = "/v14_test"
        policy_node = V14PolicyNode(namespace=namespace, parameter_overrides=[
            Parameter("actor_checkpoint", value=str(tmp_path / "actor.pt")),
            Parameter("student_checkpoint", value=str(tmp_path / "student.pt")),
            Parameter("command", value=[0.0, 0.5, 0.0]),
        ])
        command_node = V14JointCommandNode(namespace=namespace)
        sim = rclpy.create_node("fake_sim", namespace=namespace)
        qos = contract.lockstep_qos()
        state_pub = sim.create_publisher(JointState, contract.JOINT_STATES, qos)
        odom_pub = sim.create_publisher(Odometry, contract.ODOM, qos)
        image_pub = sim.create_publisher(Image, contract.DEPTH_IMAGE, qos)
        received = {}
        sim.create_subscription(JointState, contract.JOINT_COMMAND,
                                lambda m: received.__setitem__(contract.stamp_ns(m.header.stamp), list(m.position)), qos)
        executor = SingleThreadedExecutor()
        for node in (policy_node, command_node, sim):
            executor.add_node(node)
        for _ in range(200):
            if state_pub.get_subscription_count() >= 2 and image_pub.get_subscription_count() >= 1:
                break
            executor.spin_once(timeout_sec=0.05)

        # Shuffled joint order on the wire; the nodes must reorder by name.
        wire = list(reversed(contract.JOINT_NAMES))
        rng = np.random.default_rng(8)
        offline_policy = V14DepthPolicy(tmp_path / "actor.pt", tmp_path / "student.pt")
        offline_filter = JointCommandFilter()
        for step in range(12):
            sec, nanosec = contract.step_stamp(step)
            stamp = sec * 1_000_000_000 + nanosec
            position = rng.normal(size=12) * 0.2
            velocity = rng.normal(size=12)
            depth = rng.uniform(0.2, 4.0, size=(84, 84)).astype(np.float32)
            if step == 0:
                offline_filter.reset(torch.tensor(position, dtype=torch.float32))
            if contract.is_policy_tick(stamp):
                odometry = Odometry()
                odometry.header.stamp.sec, odometry.header.stamp.nanosec = sec, nanosec
                odometry.pose.pose.orientation.w = 1.0
                odometry.twist.twist.linear.y = 0.3
                image = Image(height=84, width=84, encoding="32FC1", is_bigendian=0, step=84 * 4,
                              data=depth.astype("<f4").tobytes())
                image.header.stamp = odometry.header.stamp
                odom_pub.publish(odometry)
                image_pub.publish(image)
                _, target = offline_policy.step(
                    joint_pos=position, joint_vel=velocity, lin_vel_b=(0.0, 0.3, 0.0), ang_vel_b=(0.0, 0.0, 0.0),
                    quat_wxyz=(1.0, 0.0, 0.0, 0.0), command=(0.0, 0.5, 0.0), depth_m=depth)
                offline_filter.set_target(target)
            expected = offline_filter.step(torch.tensor(position, dtype=torch.float32),
                                           torch.tensor(velocity, dtype=torch.float32))
            state = JointState(name=wire, position=list(position[::-1]), velocity=list(velocity[::-1]))
            state.header.stamp.sec, state.header.stamp.nanosec = sec, nanosec
            state_pub.publish(state)
            for _ in range(400):
                if stamp in received:
                    break
                executor.spin_once(timeout_sec=0.05)
            assert stamp in received, f"no command for physics step {step}"
            np.testing.assert_allclose(received.pop(stamp), expected.numpy(), atol=1e-5)
        assert policy_node.steps == 3 and command_node.commands == 12
    finally:
        rclpy.shutdown()
