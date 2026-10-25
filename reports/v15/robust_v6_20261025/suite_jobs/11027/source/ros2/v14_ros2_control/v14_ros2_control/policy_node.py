"""ROS2 node: V14 depth policy at 50 Hz, publishing raw joint position targets.

Runs one policy step when joint state, odometry and depth image with the same
simulation stamp have arrived (lockstep with the Isaac Sim bridge).
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from v14_ros2_control import contract
from v14_ros2_control.policy import V14DepthPolicy


def depth_from_image(message: Image) -> np.ndarray:
    if message.encoding != "32FC1" or message.is_bigendian:
        raise ValueError(f"expected little-endian 32FC1 depth, got {message.encoding}")
    if message.step != message.width * 4:
        raise ValueError("padded depth rows are not supported")
    pixels = np.frombuffer(bytes(message.data), dtype="<f4")
    return pixels.reshape(message.height, message.width).copy()


class V14PolicyNode(Node):
    def __init__(self, **node_kwargs):
        super().__init__("v14_policy", **node_kwargs)
        actor = self.declare_parameter("actor_checkpoint", "").value
        student = self.declare_parameter("student_checkpoint", "").value
        device = self.declare_parameter("device", "cpu").value
        command = self.declare_parameter("command", [0.0, 0.4, 0.0]).value
        if not actor or not student:
            raise ValueError("actor_checkpoint and student_checkpoint parameters are required")
        self.command = [float(value) for value in command]
        if len(self.command) != 3:
            raise ValueError("command must be [lin_vel_x, lin_vel_y, ang_vel_z] in the base frame")
        self.policy = V14DepthPolicy(actor, student, device=device)
        self.states, self.odometry, self.images = {}, {}, {}
        self.last_stamp = -1
        self.steps = 0
        qos = contract.lockstep_qos()
        self.target_pub = self.create_publisher(JointState, contract.JOINT_TARGET, qos)
        self.create_subscription(JointState, contract.JOINT_STATES, self.on_state, qos)
        self.create_subscription(Odometry, contract.ODOM, self.on_odometry, qos)
        self.create_subscription(Image, contract.DEPTH_IMAGE, self.on_image, qos)
        self.create_subscription(Twist, contract.CMD_VEL, self.on_cmd_vel, 10)
        self.get_logger().info(
            f"actor={actor} student={student} device={device} command={self.command}"
        )

    def on_state(self, message: JointState) -> None:
        stamp = contract.stamp_ns(message.header.stamp)
        if contract.is_policy_tick(stamp):
            self.states[stamp] = message
            self.try_step(stamp)

    def on_odometry(self, message: Odometry) -> None:
        stamp = contract.stamp_ns(message.header.stamp)
        self.odometry[stamp] = message
        self.try_step(stamp)

    def on_image(self, message: Image) -> None:
        stamp = contract.stamp_ns(message.header.stamp)
        self.images[stamp] = message
        self.try_step(stamp)

    def on_cmd_vel(self, message: Twist) -> None:
        self.command = [message.linear.x, message.linear.y, message.angular.z]

    def try_step(self, stamp: int) -> None:
        if stamp <= self.last_stamp or not all(stamp in d for d in (self.states, self.odometry, self.images)):
            return
        state, odometry, image = (d.pop(stamp) for d in (self.states, self.odometry, self.images))
        for buffer in (self.states, self.odometry, self.images):
            for old in [key for key in buffer if key < stamp]:
                del buffer[old]
        pose, twist = odometry.pose.pose, odometry.twist.twist
        q = pose.orientation
        _, target = self.policy.step(
            joint_pos=contract.ordered(state.name, state.position),
            joint_vel=contract.ordered(state.name, state.velocity),
            lin_vel_b=(twist.linear.x, twist.linear.y, twist.linear.z),
            ang_vel_b=(twist.angular.x, twist.angular.y, twist.angular.z),
            quat_wxyz=(q.w, q.x, q.y, q.z),
            command=self.command,
            depth_m=depth_from_image(image),
        )
        message = JointState()
        message.header.stamp = state.header.stamp
        message.name = list(contract.JOINT_NAMES)
        message.position = target.tolist()
        self.target_pub.publish(message)
        self.last_stamp = stamp
        self.steps += 1
        if self.steps % 500 == 0:
            self.get_logger().info(f"policy steps={self.steps} sim_time={stamp * 1e-9:.2f}s")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = V14PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
