"""V15 depth locomotion only, independent of the old WBC physical model."""
import json
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState,Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from v14_ros2_control import contract
from v14_ros2_control.policy import V14DepthPolicy
from v14_ros2_control.policy_node import depth_from_image
from v14_ros2_control.joint_command import JointCommandFilter
from .motion import CANONICAL,hybrid_motor_command

class V15WalkNode(Node):
    def __init__(self,**kwargs):
        super().__init__('v15_walk_controller',**kwargs)
        param=lambda n,d:self.declare_parameter(n,d).value
        torch.set_num_threads(1)
        self.policy=V14DepthPolicy(param('actor_checkpoint',''),param('student_checkpoint',''))
        self.base_com_offset=np.asarray(param('base_com_offset',[0.,0.,0.]),float)
        if self.base_com_offset.shape!=(3,) or not np.isfinite(self.base_com_offset).all():
            raise ValueError('base_com_offset must contain three finite meters')
        self.command=[0.,float(param('speed',.4)),0.]
        self.filter=JointCommandFilter();self.initialized=False;self.policy_steps=0
        self.states={};self.odoms={};self.images={};self.contacts={}
        self.last=-1;self.reply=None;self.support_feet=0
        self.to_canonical=[contract.JOINT_NAMES.index(n) for n in CANONICAL]
        self.to_policy=[CANONICAL.index(n) for n in contract.JOINT_NAMES]
        qos=contract.lockstep_qos()
        self.pub=self.create_publisher(JointState,'joint_command',qos)
        self.status_pub=self.create_publisher(String,'status',qos)
        self.create_subscription(JointState,'joint_states',self.on_state,qos)
        for typ,topic,buffer in [(Odometry,'odom',self.odoms),(Image,'depth/image',self.images),
                                 (JointState,'foot_contacts',self.contacts)]:
            self.create_subscription(typ,topic,lambda m,b=buffer:self.add(b,m),qos)
        self.create_subscription(Twist,'cmd_vel',self.on_command,10)
        self.get_logger().info('V15 WALK ready; depth policy, +y forward, 200/50 Hz lockstep')

    def on_command(self,m):
        command=[m.linear.x,m.linear.y,m.angular.z]
        if not np.isfinite(command).all():raise ValueError('Nonfinite walking command')
        self.command=command

    def on_state(self,m):
        if contract.stamp_ns(m.header.stamp)==self.last and self.reply is not None:
            self.pub.publish(self.reply[0]);self.status_pub.publish(self.reply[1]);return
        self.add(self.states,m)

    def add(self,buffer,m):
        stamp=contract.stamp_ns(m.header.stamp)
        if stamp>self.last:buffer[stamp]=m;self.drain()

    def drain(self):
        for stamp in sorted(self.states):
            if stamp<=self.last:continue
            tick=contract.is_policy_tick(stamp)
            if tick and not all(stamp in d for d in (self.odoms,self.images,self.contacts)):break
            start=time.perf_counter();state=self.states.pop(stamp)
            q=np.asarray(contract.ordered(state.name,state.position));dq=np.asarray(contract.ordered(state.name,state.velocity))
            qt=torch.tensor(q,dtype=torch.float32);dqt=torch.tensor(dq,dtype=torch.float32)
            if not self.initialized:self.filter.reset(qt);self.initialized=True
            if tick:
                odom=self.odoms.pop(stamp);image=self.images.pop(stamp);contacts=self.contacts.pop(stamp)
                pose=odom.pose.pose;twist=odom.twist.twist
                quat=[pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z]
                vel=np.array([twist.linear.x,twist.linear.y,twist.linear.z])
                omega=np.array([twist.angular.x,twist.angular.y,twist.angular.z])
                _,target=self.policy.step(joint_pos=q,joint_vel=dq,lin_vel_b=vel+np.cross(omega,self.base_com_offset),
                    ang_vel_b=omega,quat_wxyz=quat,command=self.command,depth_m=depth_from_image(image))
                self.filter.set_target(target);self.policy_steps+=1
                self.support_feet=sum(f>5. for f in contacts.effort)
            target=self.filter.step(qt,dqt).numpy()
            motor=hybrid_motor_command(1.,np.zeros(12),target[self.to_canonical],q[self.to_canonical],dq[self.to_canonical])
            message=JointState();message.header=state.header;message.name=list(contract.JOINT_NAMES)
            message.position=motor['position'][self.to_policy].tolist();message.effort=motor['effort'][self.to_policy].tolist()
            status=String();status.data=json.dumps(dict(stamp_ns=stamp,mode='WALK',walk_weight=1.,height_fraction=1.,
                command=self.command,command_schema=2,support_feet=self.support_feet,wbc={'active':False},
                requested_torque=motor['requested_torque'].tolist(),bounded_torque=motor['bounded_torque'].tolist(),
                stiffness_nm_rad=motor['kp'],damping_nms_rad=motor['kd'],base_com_offset_m=self.base_com_offset.tolist(),
                compute_ms=1000*(time.perf_counter()-start)),allow_nan=False)
            self.last=stamp;self.reply=(message,status)
            self.pub.publish(message);self.status_pub.publish(status)
            if stamp==0:self.get_logger().info('Initial depth sensor set processed and motor command published')
            for buffer in (self.states,self.odoms,self.images,self.contacts):
                for old in [s for s in buffer if s<=stamp]:buffer.pop(old,None)

def main():
    rclpy.init();node=V15WalkNode()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
