"""Align fresh measured sensors to MCU joint feedback without faking missing data."""
import copy
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState,Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from v14_ros2_control import contract
from .core import SensorGate


class SensorSync(Node):
    def __init__(self,**kwargs):
        kwargs.setdefault('namespace','/v15real')
        super().__init__('v15_sensor_sync',**kwargs)
        self.gate=SensorGate();self.samples={};self.failed=False
        qos=contract.lockstep_qos()
        self.pubs={name:self.create_publisher(kind,topic,qos) for name,kind,topic in
            [('joints',JointState,'joint_states'),('odom',Odometry,'odom'),
             ('depth',Image,'depth/image'),('contacts',JointState,'foot_contacts')]}
        self.policy_joints_pub=self.create_publisher(JointState,'policy_joint_states',qos)
        self.fault_pub=self.create_publisher(String,'hardware_fault',10)
        self.create_subscription(JointState,'raw/joint_states',self.on_joint,10)
        for name,kind,topic in [('odom',Odometry,'raw/odom'),('depth',Image,'raw/depth/image'),
                              ('contacts',JointState,'raw/foot_contacts')]:
            self.create_subscription(kind,topic,lambda m,n=name:self.on_sensor(n,m),qos_profile_sensor_data)

    def fail(self,reason):
        if self.failed:return
        self.failed=True;message=String();message.data=str(reason)
        self.fault_pub.publish(message);self.get_logger().error(str(reason))

    def on_sensor(self,name,message):
        if self.failed:return
        try:
            if name=='depth':
                if (message.height,message.width,message.encoding,message.step,message.is_bigendian)!=(84,84,'32FC1',336,0) or len(message.data)!=84*84*4:
                    raise ValueError('depth must be little-endian 84x84 metric 32FC1')
            elif name=='contacts':
                names=[f'{leg}_foot' for leg in ('FL','FR','RL','RR')]
                values=contract.ordered(message.name,message.effort,expected=names)
                if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid measured foot force')
                message=copy.deepcopy(message);message.name=names;message.effort=values
            else:
                q=message.pose.pose.orientation;v=message.twist.twist;p=message.pose.pose.position
                values=[q.w,q.x,q.y,q.z,p.x,p.y,p.z,v.linear.x,v.linear.y,v.linear.z,v.angular.x,v.angular.y,v.angular.z]
                if any(not math.isfinite(x) for x in values) or abs(sum(x*x for x in values[:4])-1)>1e-3:
                    raise ValueError('odom requires finite values and normalized quaternion')
                if message.child_frame_id!='base':raise ValueError('odom body-frame twist must use child_frame_id=base')
            self.samples[name]=message
        except (KeyError,TypeError,ValueError) as error:self.fail(error)

    @staticmethod
    def stamped(message,stamp):
        result=copy.deepcopy(message)
        result.header.stamp.sec,result.header.stamp.nanosec=divmod(stamp,1_000_000_000)
        return result

    def on_joint(self,message):
        if self.failed:return
        try:
            for values in (message.position,message.velocity,message.effort):
                ordered=contract.ordered(message.name,values)
                if any(not math.isfinite(x) for x in ordered):raise ValueError('invalid measured joints')
            measurement={name:contract.stamp_ns(m.header.stamp) for name,m in self.samples.items()}
            stamp=self.gate.step(contract.stamp_ns(message.header.stamp),self.get_clock().now().nanoseconds,measurement)
            if stamp is None:return
            if contract.is_policy_tick(stamp):
                for name in ('odom','depth','contacts'):self.pubs[name].publish(self.stamped(self.samples[name],stamp))
                self.policy_joints_pub.publish(self.stamped(message,stamp))
            self.pubs['joints'].publish(self.stamped(message,stamp))
        except (KeyError,TypeError,ValueError) as error:self.fail(error)


def main(args=None):
    rclpy.init(args=args);node=SensorSync()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
