"""Isaac-only transport adapter; the controller lives in a separate ROS2 process."""
import array
import json
import time
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from v14_ros2_bridge import V14Ros2JointTargetAction, V14Ros2JointTargetActionCfg
from v14_ros2_control import contract

class V151Ros2Action(V14Ros2JointTargetAction):
    def _start_ros(self,cfg):
        import rclpy
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image,JointState
        from std_msgs.msg import String
        if not rclpy.ok(): rclpy.init()
        self._msg=dict(Clock=Clock,Image=Image,JointState=JointState,Odometry=Odometry)
        self._node=rclpy.create_node('v151_isaac_adapter',namespace=cfg.namespace)
        qos=contract.lockstep_qos()
        self._state_pub=self._node.create_publisher(JointState,contract.JOINT_STATES,qos)
        self._odom_pub=self._node.create_publisher(Odometry,contract.ODOM,qos)
        self._image_pub=self._node.create_publisher(Image,contract.DEPTH_IMAGE,qos)
        self._contact_pub=self._node.create_publisher(JointState,'foot_contacts',qos)
        self._clock_pub=self._node.create_publisher(Clock,'/clock',10)
        command_sub=self._node.create_subscription(JointState,contract.JOINT_COMMAND,self._on_command,qos)
        self._statuses={}; self.history=[]; self.drive_gain_checks=[]
        self._node.create_subscription(String,'status',self._on_status,qos)
        deadline=time.monotonic()+cfg.startup_timeout_s
        while not (self._state_pub.get_subscription_count()>=1 and self._odom_pub.get_subscription_count()>=1
            and self._image_pub.get_subscription_count()>=1 and command_sub.get_publisher_count()>=1):
            if time.monotonic()>deadline: raise TimeoutError('V15.1 ROS2 controller did not connect')
            rclpy.spin_once(self._node,timeout_sec=.1)

    def _on_command(self,message):
        self._commands[contract.stamp_ns(message.header.stamp)]=(
            contract.ordered(message.name,message.position),contract.ordered(message.name,message.effort))

    def _on_status(self,msg):
        data=json.loads(msg.data); self._statuses[data['stamp_ns']]=data

    def _publish_policy_inputs(self,sec,nanosec):
        data=self._asset.data
        odom=self._header(self._msg['Odometry'](),sec,nanosec,'world'); odom.child_frame_id='base'
        pose=odom.pose.pose; twist=odom.twist.twist
        pose.position.x,pose.position.y,pose.position.z=data.root_link_pos_w[0].tolist()
        pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z=data.root_link_quat_w[0].tolist()
        linear=quat_apply_inverse(data.root_link_quat_w,data.root_link_lin_vel_w)[0]
        twist.linear.x,twist.linear.y,twist.linear.z=linear.tolist()
        twist.angular.x,twist.angular.y,twist.angular.z=data.root_ang_vel_b[0].tolist()
        self._odom_pub.publish(odom)
        depth=self._env.scene['depth_camera'].data.output['distance_to_image_plane'][0,:,:,0]
        msg=self._header(self._msg['Image'](),sec,nanosec,'front_depth')
        msg.height,msg.width=depth.shape; msg.encoding='32FC1';msg.is_bigendian=0;msg.step=depth.shape[1]*4
        msg.data=array.array('B',depth.float().cpu().numpy().astype('<f4').tobytes());self._image_pub.publish(msg)
        sensor=self._env.scene['contact_forces']; ids,_=sensor.find_bodies(['FL_foot','FR_foot','RL_foot','RR_foot'],preserve_order=True)
        contact=self._header(self._msg['JointState'](),sec,nanosec,'world')
        contact.name=['FL_foot','FR_foot','RL_foot','RR_foot']
        contact.effort=torch.linalg.vector_norm(sensor.data.net_forces_w[0,ids],dim=-1).tolist()
        self._contact_pub.publish(contact)

    def apply_actions(self):
        import rclpy
        sec,nanosec=contract.step_stamp(self._physics_step); stamp=sec*1_000_000_000+nanosec
        if contract.is_policy_tick(stamp): self._publish_policy_inputs(sec,nanosec)
        self._publish_joint_state(sec,nanosec)
        deadline=time.monotonic()+self.cfg.step_timeout_s
        while stamp not in self._commands or stamp not in self._statuses:
            if time.monotonic()>deadline: raise TimeoutError(f'V15.1 command missing at {stamp*1e-9:.3f}s')
            rclpy.spin_once(self._node,timeout_sec=.001)
        position,effort=self._commands.pop(stamp)
        status=self._statuses.pop(stamp)
        if status.get('command_schema')!=2: raise ValueError('V15.1 hybrid command schema mismatch')
        kp,kd=float(status['stiffness_nm_rad']),float(status['damping_nms_rad'])
        if not (0<=kp<=300 and 0<=kd<=8): raise ValueError('invalid drive gains')
        if getattr(self,'_drive_gains',None)!=(kp,kd):
            self._asset.write_joint_stiffness_to_sim(kp)
            self._asset.write_joint_damping_to_sim(kd)
            for actuator in self._asset.actuators.values():
                actuator.stiffness[:]=kp;actuator.damping[:]=kd
            if kp in (0.,300.):
                actual_kp=self._asset.root_physx_view.get_dof_stiffnesses().cpu()
                actual_kd=self._asset.root_physx_view.get_dof_dampings().cpu()
                if not torch.allclose(actual_kp,torch.full_like(actual_kp,kp)) or not torch.allclose(actual_kd,torch.full_like(actual_kd,kd)):
                    raise RuntimeError('PhysX gains disagree with ROS2 command')
                self.drive_gain_checks.append(dict(stamp_ns=stamp,kp=actual_kp.tolist(),kd=actual_kd.tolist()))
            self._drive_gains=(kp,kd)
        command=torch.tensor(position,device=self.device)
        feedforward=torch.tensor(effort,device=self.device)[None]
        if not torch.isfinite(feedforward).all(): raise ValueError('nonfinite feedforward')
        if command.shape!=(12,) or not torch.isfinite(command).all(): raise ValueError('invalid ROS2 target')
        self._processed_actions[0]=command
        self._asset.set_joint_position_target(self._processed_actions,joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(feedforward,joint_ids=self._joint_ids)
        self.history.append(status);self._physics_step+=1

    def reset(self,env_ids=None):
        if self._physics_step:
            raise RuntimeError('V15.1 evaluation forbids robot reset after the first command')

@configclass
class V151Ros2ActionCfg(V14Ros2JointTargetActionCfg):
    class_type:type=V151Ros2Action
    namespace:str='/v151'
