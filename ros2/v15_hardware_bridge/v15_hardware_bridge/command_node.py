"""200 Hz command process; planner failure expires rather than blocking this node."""
import json
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import torch
from v14_ros2_control import contract
from .command_stage import PlanCommandStage


class JointCommandNode(Node):
    def __init__(self,**kwargs):
        super().__init__('v15_joint_command',**kwargs)
        torch.set_num_threads(1)
        # Initialize filter and motor-math kernels with disposable state before
        # subscriptions exist. No synthetic command is published.
        warm=PlanCommandStage()
        q=[contract.joint_default(n) for n in contract.JOINT_NAMES]
        warm.update(dict(plan_schema=1,source_stamp_ns=0,generation=0,raw_target=q,
            standing_torque=[0.]*12,status=dict(mode='STAND',walk_weight=0.,
            height_fraction=1.,command=[0.,0.,0.])),0)
        for step in range(4):warm.command(step*5_000_000,contract.JOINT_NAMES,q,[0.]*12,step*5_000_000)
        self.stage=PlanCommandStage();self.failed=False
        qos=contract.lockstep_qos()
        self.command_pub=self.create_publisher(JointState,'joint_command',qos)
        self.status_pub=self.create_publisher(String,'status',qos)
        self.fault_pub=self.create_publisher(String,'hardware_fault',10)
        self.create_subscription(String,'control_plan',self.on_plan,qos)
        self.create_subscription(JointState,'joint_states',self.on_state,qos)
        self.create_subscription(String,'hardware_fault',lambda m:self.fail(m.data),10)
        self.create_timer(.01,self.check_deadline)

    def fail(self,reason):
        if self.failed:return
        self.failed=True
        message=String(data=str(reason));self.fault_pub.publish(message)
        self.get_logger().error(str(reason))

    def on_plan(self,message):
        if self.failed:return
        try:self.stage.update(json.loads(message.data),time.monotonic_ns())
        except (KeyError,TypeError,ValueError,OverflowError) as error:self.fail(error)

    def check_deadline(self):
        if self.failed:return
        try:self.stage.check_deadline(time.monotonic_ns())
        except (ValueError,TimeoutError) as error:self.fail(error)

    def on_state(self,message):
        if self.failed:return
        try:
            start=time.perf_counter()
            result=self.stage.command(contract.stamp_ns(message.header.stamp),message.name,
                                      message.position,message.velocity,time.monotonic_ns())
            if result is None:return
            command=JointState();command.header=message.header;command.name=result['names']
            command.position=result['position'];command.effort=result['effort']
            status=dict(result['status'],compute_ms=1000*(time.perf_counter()-start))
            self.stage.check_deadline(time.monotonic_ns())
            self.command_pub.publish(command)
            self.status_pub.publish(String(data=json.dumps(status,allow_nan=False)))
        except (KeyError,TypeError,ValueError,RuntimeError,TimeoutError) as error:self.fail(error)


def main(args=None):
    rclpy.init(args=args);node=JointCommandNode()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
