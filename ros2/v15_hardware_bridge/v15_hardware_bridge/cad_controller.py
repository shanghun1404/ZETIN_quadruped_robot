"""Deployment entry point for the existing CAD controller, with operator timeout."""
import time
import rclpy
from v151_ros2_control import controller_node
from v151_ros2_control.cad_standing import CadStandingController


class HardwareController(controller_node.ControllerNode):
    def __init__(self,**kwargs):
        controller_node.StandingController=CadStandingController
        super().__init__(**kwargs)
        self.command_time=None
        self.operator_timeout=float(self.declare_parameter('cmd_vel_timeout_s',.5).value)
        if self.operator_timeout<=0:raise ValueError('operator timeout must be positive')
        self.operator_timer=self.create_timer(.05,self.check_operator)

    def command(self,message):
        self.command_time=time.monotonic()
        super().command(message)

    def check_operator(self):
        if self.supervisor.requested=='walk' and (self.command_time is None or time.monotonic()-self.command_time>self.operator_timeout):
            self.request_command=[0.,0.,0.]
            self.supervisor.request('stand',self.request_command)


def main(args=None):
    rclpy.init(args=args);node=HardwareController()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
