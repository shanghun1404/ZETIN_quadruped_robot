"""ROS2 node: 200 Hz V13 LPF + torque-envelope guard between policy and joint drives."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from v14_ros2_control import contract
from v14_ros2_control.joint_command import JointCommandFilter, LockstepJointCommander


class V14JointCommandNode(Node):
    def __init__(self, **node_kwargs):
        super().__init__("v14_joint_command", **node_kwargs)
        cutoff_hz = self.declare_parameter("cutoff_hz", 25.0).value
        guard = self.declare_parameter("guard_fraction", 0.99).value
        guard = None if guard <= 0.0 else guard
        self.commander = LockstepJointCommander(JointCommandFilter(cutoff_hz=cutoff_hz, guard_fraction=guard))
        self.stamps = {}
        self.commands = 0
        qos = contract.lockstep_qos()
        self.command_pub = self.create_publisher(JointState, contract.JOINT_COMMAND, qos)
        self.create_subscription(JointState, contract.JOINT_STATES, self.on_state, qos)
        self.create_subscription(JointState, contract.JOINT_TARGET, self.on_target, qos)
        self.get_logger().info(f"cutoff_hz={cutoff_hz} guard_fraction={guard}")

    def on_state(self, message: JointState) -> None:
        stamp = contract.stamp_ns(message.header.stamp)
        self.stamps[stamp] = message.header.stamp
        self.commander.add_state(
            stamp,
            contract.ordered(message.name, message.position),
            contract.ordered(message.name, message.velocity),
        )
        self.publish_ready()

    def on_target(self, message: JointState) -> None:
        stamp = contract.stamp_ns(message.header.stamp)
        self.commander.add_target(stamp, contract.ordered(message.name, message.position))
        self.publish_ready()

    def publish_ready(self) -> None:
        for stamp, command in self.commander.drain():
            message = JointState()
            message.header.stamp = self.stamps.pop(stamp)
            message.name = list(contract.JOINT_NAMES)
            message.position = command.tolist()
            self.command_pub.publish(message)
            self.commands += 1
            if self.commands % 2000 == 0:
                self.get_logger().info(f"joint commands={self.commands} sim_time={stamp * 1e-9:.2f}s")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = V14JointCommandNode()
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
