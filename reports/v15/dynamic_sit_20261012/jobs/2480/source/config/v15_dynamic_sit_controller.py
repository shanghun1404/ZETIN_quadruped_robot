"""Test-only ROS2 request schedule; physics advances only via motor commands."""
import rclpy
from v14_ros2_control import contract
import v151_ros2_control.controller_node as controller
from v151_ros2_control.cad_standing import CadStandingController

controller.StandingController=CadStandingController


class DynamicSitDemoNode(controller.ControllerNode):
    def add(self,buffer,message):
        stamp=contract.stamp_ns(message.header.stamp)
        if stamp>self.last:
            t=stamp*1e-9
            self.supervisor.request('sit' if 2.<=t<8. else 'stand')
        super().add(buffer,message)


def main():
    rclpy.init();node=DynamicSitDemoNode()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
