"""Test-only ROS2 entry point; schedules are driven by received sensor stamps."""
import os
import rclpy
from v14_ros2_control import contract
from v15_cad_suite_protocol import command_at, get_case
from v151_ros2_control.v15_walk_node import V15WalkNode


class ProfileWalkNode(V15WalkNode):
    def __init__(self):
        self.case=get_case(os.environ['V15_CAD_CASE'])
        super().__init__()

    def add(self,buffer,message):
        stamp=contract.stamp_ns(message.header.stamp)
        if stamp>self.last:
            self.command=list(command_at(self.case,stamp*1e-9))
        super().add(buffer,message)


def main():
    rclpy.init()
    case=os.environ['V15_CAD_CASE']
    if case in ('standing_hold','sit_rise_walk_stop_sit','deep_crouch_rise_walk_stop_sit'):
        import v151_ros2_control.controller_node as controller
        from v151_ros2_control.cad_standing import CadStandingController
        # The existing supervisor and ROS2 motor transport run unchanged.
        controller.StandingController=CadStandingController
        node=controller.ControllerNode()
    else:
        node=ProfileWalkNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
