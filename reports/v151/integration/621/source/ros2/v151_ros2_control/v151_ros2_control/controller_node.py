"""V15.1 ROS2 controller. Only sensor messages enter; no simulator API is imported."""
import json
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from v14_ros2_control import contract
from v14_ros2_control.policy import V14DepthPolicy, projected_gravity
from v14_ros2_control.policy_node import depth_from_image
from v14_ros2_control.joint_command import JointCommandFilter
from .motion import MotionSupervisor, CANONICAL, hybrid_motor_command, posture_forward_offset
from .standing import StandingController

class ControllerNode(Node):
    def __init__(self):
        super().__init__('v151_controller')
        param=lambda n,d:self.declare_parameter(n,d).value
        torch.set_num_threads(1)
        self.policy=V14DepthPolicy(param('actor_checkpoint',''),param('student_checkpoint',''),device='cpu')
        self.standing=StandingController(param('model_path',''),dt=.02)
        self.base_com_offset=np.array(next(r['com_body'] for r in self.standing.model.body_records if r['name']=='base'))
        self.initial=param('initial_mode','SIT')
        self.supervisor=MotionSupervisor(initial_mode=self.initial)
        self.demo=param('demo',False)
        self.speed=float(param('speed',.4))
        self.drop=float(param('height_drop',.10))
        if self.initial=='WALK': self.supervisor.request('walk',[0.,self.speed,0.])
        self.filter=JointCommandFilter()
        self.states={}; self.odoms={}; self.images={}; self.contacts={}
        self.last=-1; self.initialized=False; self.target=None; self.stand_tau=np.zeros(12)
        self.output=dict(mode=self.initial,walk_weight=0.,height_fraction=0.,command=[0.,0.,0.])
        self.nominal=np.tile([0.,-.55,1.1],4)
        self.forward_shift=float(param('stand_forward_shift_m',.03))
        self.crouch_forward_shift=float(param('crouch_forward_shift_m',0.))
        if not np.isfinite(self.crouch_forward_shift):
            raise ValueError('crouch_forward_shift_m must be finite')
        self.stand_reference=self.standing.crouch_pose(0.,forward_shift_m=self.forward_shift)[0]
        self.to_canonical=[contract.JOINT_NAMES.index(n) for n in CANONICAL]
        self.to_policy=[CANONICAL.index(n) for n in contract.JOINT_NAMES]
        qos=contract.lockstep_qos()
        self.pub=self.create_publisher(JointState,contract.JOINT_COMMAND,qos)
        self.status_pub=self.create_publisher(String,'status',qos)
        self.create_subscription(JointState,contract.JOINT_STATES,self.on_state,qos)
        self.create_subscription(Odometry,contract.ODOM,lambda m:self.add(self.odoms,m),qos)
        self.create_subscription(Image,contract.DEPTH_IMAGE,lambda m:self.add(self.images,m),qos)
        self.create_subscription(JointState,'foot_contacts',lambda m:self.add(self.contacts,m),qos)
        self.create_subscription(Twist,contract.CMD_VEL,self.command,10)
        self.create_subscription(String,'mode_request',self.mode_request,10)
        self.request_command=[0.,0.,0.]
        self.get_logger().info('V15.1 ROS2 controller ready; +y forward; lockstep 200/50 Hz')

    def command(self,m):
        self.request_command=[m.linear.x,m.linear.y,m.angular.z]
        mode='walk' if np.linalg.norm(self.request_command)>1e-5 else 'stand'
        self.supervisor.request(mode,self.request_command)

    def mode_request(self,m):
        self.supervisor.request(m.data,self.request_command)

    def add(self,buffer,m):
        stamp=contract.stamp_ns(m.header.stamp)
        if stamp>self.last:
            buffer[stamp]=m
            if stamp==0 and self.last<0:
                self.get_logger().info('Initial input received: '+type(m).__name__+
                    f'; state/odom/image/contact={len(self.states)}/{len(self.odoms)}/{len(self.images)}/{len(self.contacts)}')
            self.drain()

    def on_state(self,m): self.add(self.states,m)

    def drain(self):
        for stamp in sorted(self.states):
            if stamp<=self.last: continue
            tick=contract.is_policy_tick(stamp)
            if tick and not all(stamp in d for d in (self.odoms,self.images,self.contacts)): break
            if stamp==0: self.get_logger().info('Initial synchronized sensor set ready')
            start=time.perf_counter()
            state=self.states.pop(stamp)
            q=np.array(contract.ordered(state.name,state.position)); dq=np.array(contract.ordered(state.name,state.velocity))
            qc=q[self.to_canonical]; dqc=dq[self.to_canonical]
            if not self.initialized:
                self.filter.reset(torch.tensor(q,dtype=torch.float32))
                self.crouch=qc.copy() if self.initial=='SIT' else self.standing.crouch_pose(self.drop)[0]
                self.initialized=True
            if tick:
                odom=self.odoms.pop(stamp); image=self.images.pop(stamp); contact=self.contacts.pop(stamp)
                pose=odom.pose.pose; twist=odom.twist.twist
                xyz=np.array([pose.position.x,pose.position.y,pose.position.z])
                quat=np.array([pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z])
                vel=np.array([twist.linear.x,twist.linear.y,twist.linear.z]); omega=np.array([twist.angular.x,twist.angular.y,twist.angular.z])
                if not hasattr(self,'stand_height'):
                    self.sit_height=xyz[2] if self.initial=='SIT' else xyz[2]-self.drop
                    self.stand_height=self.sit_height+self.drop
                t=stamp*1e-9
                if not hasattr(self,'reference_start_s'): self.reference_start_s=t
                if self.demo:
                    if t<.5: self.supervisor.request('sit' if self.initial=='SIT' else 'stand')
                    elif t<7.: self.supervisor.request('stand')
                    elif t<17.: self.supervisor.request('walk',[0.,self.speed,0.])
                    elif t<23.: self.supervisor.request('stand')
                    else: self.supervisor.request('sit')
                gravity=projected_gravity(torch.tensor(quat)).numpy()
                stable=(gravity[2]<-.97 and np.linalg.norm(vel)<.12 and np.linalg.norm(omega)<.35
                        and sum(f>5. for f in contact.effort)==4 and abs(xyz[2]-self.stand_height)<.035)
                support_mask=np.asarray(contact.effort)>5.
                previous=self.output['mode']
                self.output=self.supervisor.step(t,stable=stable,support_ready=(gravity[2]<-.97 and support_mask.sum()>=2),seated_ready=(gravity[2]<-.97 and np.linalg.norm(vel)<.12
                    and np.linalg.norm(omega)<.35 and sum(f>5. for f in contact.effort)==4
                    and abs(xyz[2]-self.sit_height)<.035))
                weight=self.output['walk_weight']
                if self.output['mode']!=previous:
                    self.get_logger().info(f"{t:.2f}s {previous} -> {self.output['mode']}")
                    if self.output['mode']=='BLEND_WALK':
                        self.filter.reset(torch.tensor(q,dtype=torch.float32)); self.policy.reset()
                    if self.output['mode']=='BLEND_STAND':
                        # Anchors/reference must follow the robot's new location after walking.
                        self.standing=StandingController(self.standing.model.source_path,dt=.02)
                        self.standing.initialize(qc,xyz,quat,stance_mask=support_mask)
                        self.handover_q=qc.copy(); self.handover_height=float(xyz[2]); self.measured_support=True
                h=self.sit_height+(self.stand_height-self.sit_height)*self.output['height_fraction']
                qref=self.crouch+(self.stand_reference-self.crouch)*self.output['height_fraction']
                forward_offset=self.forward_shift*self.output['height_fraction']
                if self.crouch_forward_shift != 0.:
                    forward_offset=posture_forward_offset(self.output['height_fraction'],self.forward_shift,
                        self.crouch_forward_shift,t-self.reference_start_s)
                if self.output['mode']=='BLEND_STAND':
                    qref=weight*self.handover_q+(1-weight)*self.stand_reference
                    forward_offset=(1-weight)*self.forward_shift
                    h=weight*self.handover_height+(1-weight)*self.stand_height
                diag={}
                if weight<1.:
                    body_reference=None
                    if getattr(self,'measured_support',False) or self.crouch_forward_shift != 0.:
                        reference_support=support_mask if getattr(self,'measured_support',False) else np.ones(4,dtype=bool)
                        qref,body_reference=self.standing.contact_consistent_reference(qc,xyz,quat,h,reference_support,
                            center_body=(1-weight) if self.output['mode']=='BLEND_STAND' else 1.,forward_offset_m=forward_offset)
                    self.stand_tau,diag=self.standing.compute(qc,dqc,xyz,quat,vel,omega,height=h,joint_reference=qref,
                        stance_mask=support_mask if getattr(self,'measured_support',False) else None,forward_offset_m=forward_offset,
                        position_reference_world=body_reference)
                if weight>0.:
                    _,target=self.policy.step(joint_pos=q,joint_vel=dq,lin_vel_b=vel+np.cross(omega,self.base_com_offset),ang_vel_b=omega,
                        quat_wxyz=quat,command=self.output['command'],depth_m=depth_from_image(image))
                    self.filter.set_target(target)
                else:
                    self.filter.reset(torch.tensor(q,dtype=torch.float32))
                self.support_feet=sum(f>5. for f in contact.effort)
                self.last_diag={k:(v.item() if isinstance(v,np.generic) else v) for k,v in diag.items()
                    if isinstance(v,(bool,int,float,np.generic))}
                self.last_diag['active']=weight<1.
                self.last_vectors={k:np.asarray(diag[k]).tolist() for k in
                    ('base_position_reference_world','forces_world','acceleration','stance_mask') if k in diag}
                if weight<1.: self.last_vectors['joint_reference']=np.asarray(qref).tolist()
            weight=self.output['walk_weight']
            walk_target=self.filter.step(torch.tensor(q,dtype=torch.float32),torch.tensor(dq,dtype=torch.float32)).numpy()
            motor=hybrid_motor_command(weight,self.stand_tau,walk_target[self.to_canonical],qc,dqc)
            target=motor['position'];bounded=motor['bounded_torque'];requested=motor['requested_torque']
            if self.output['mode']=='FAULT':
                raise RuntimeError('V15.1 readiness timeout: transition failed')
            message=JointState(); message.header=state.header; message.name=list(contract.JOINT_NAMES)
            message.position=target[self.to_policy].tolist(); message.effort=motor['effort'][self.to_policy].tolist()
            self.pub.publish(message)
            status=String(); status.data=json.dumps(dict(stamp_ns=stamp,**self.output,compute_ms=1000*(time.perf_counter()-start),
                requested_torque=requested.tolist(),bounded_torque=bounded.tolist(),
                support_feet=self.support_feet,wbc=self.last_diag,wbc_vectors=self.last_vectors,command_schema=2,stand_forward_shift_m=self.forward_shift,
                crouch_forward_shift_m=self.crouch_forward_shift,
                stiffness_nm_rad=motor['kp'],damping_nms_rad=motor['kd']),allow_nan=False)
            self.status_pub.publish(status)
            if stamp==0: self.get_logger().info('Initial command and status published')
            self.last=stamp
            for buffer in (self.states,self.odoms,self.images,self.contacts):
                for old in [s for s in buffer if s<=stamp]: buffer.pop(old,None)

def main(args=None):
    rclpy.init(args=args); node=ControllerNode()
    try: rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__=='__main__': main()
