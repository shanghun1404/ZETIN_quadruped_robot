"""Isaac adapter for the CAD USD, reusing the validated ROS2 lockstep transport."""
import array
from types import SimpleNamespace
import torch
from isaaclab.utils.math import quat_apply_inverse
from v14_ros2_control import contract
from v151_ros2_bridge import V151Ros2Action
from v15_cad_contract import joint_mapping,physical_command_fields

class V15CadBridge(V151Ros2Action):
    def __init__(self,scene,device,namespace='/v15cad'):
        self.cfg=SimpleNamespace(namespace=namespace,startup_timeout_s=60.,step_timeout_s=30.)
        self._env=SimpleNamespace(scene=scene,device=device)
        self._asset=scene['robot'];self._debug_vis_handle=None
        names,sign=joint_mapping(contract.JOINT_NAMES)
        self._joint_ids,found=self._asset.find_joints(list(names),preserve_order=True)
        if tuple(found)!=names:raise ValueError('CAD joint mapping does not cover articulation')
        self.sign=torch.tensor(sign,device=device,dtype=torch.float32)
        self._processed_actions=torch.zeros(1,12,device=device)
        self._raw_actions=self._processed_actions.clone()
        self._physics_step=0;self._commands={}
        self._start_ros(self.cfg)

    def _on_command(self,message):
        super()._on_command(message)
        stamp=contract.stamp_ns(message.header.stamp)
        if stamp in self._commands:
            q,tau=self._commands[stamp]
            sign=self.sign.cpu().numpy()
            self._commands[stamp]=physical_command_fields(q,tau,sign)

    def _publish_joint_state(self,sec,nanosec):
        data=self._asset.data
        message=self._header(self._msg['JointState'](),sec,nanosec)
        message.name=list(contract.JOINT_NAMES)
        message.position=(data.joint_pos[0,self._joint_ids]*self.sign).tolist()
        message.velocity=(data.joint_vel[0,self._joint_ids]*self.sign).tolist()
        message.effort=(data.applied_torque[0,self._joint_ids]*self.sign).tolist()
        self._publish_input(self._state_pub,message)
        clock=self._msg['Clock']();clock.clock.sec,clock.clock.nanosec=sec,nanosec
        self._clock_pub.publish(clock)

    def _publish_policy_inputs(self,sec,nanosec):
        data=self._asset.data
        odom=self._header(self._msg['Odometry'](),sec,nanosec,'world');odom.child_frame_id='base'
        pose=odom.pose.pose;twist=odom.twist.twist
        pose.position.x,pose.position.y,pose.position.z=data.root_link_pos_w[0].tolist()
        pose.orientation.w,pose.orientation.x,pose.orientation.y,pose.orientation.z=data.root_link_quat_w[0].tolist()
        twist.linear.x,twist.linear.y,twist.linear.z=quat_apply_inverse(data.root_link_quat_w,data.root_link_lin_vel_w)[0].tolist()
        twist.angular.x,twist.angular.y,twist.angular.z=data.root_ang_vel_b[0].tolist()
        self._publish_input(self._odom_pub,odom)
        depth=self._env.scene['depth_camera'].data.output['distance_to_image_plane'][0,:,:,0]
        msg=self._header(self._msg['Image'](),sec,nanosec,'front_depth')
        msg.height,msg.width=depth.shape;msg.encoding='32FC1';msg.is_bigendian=0;msg.step=msg.width*4
        msg.data=array.array('B',depth.float().cpu().numpy().astype('<f4').tobytes())
        self._publish_input(self._image_pub,msg)
        contact=self._header(self._msg['JointState'](),sec,nanosec,'world')
        contact.name=['FL_foot','FR_foot','RL_foot','RR_foot']
        # CAD feet are meshes on each shank, not separate rigid bodies.
        bodies=['FL_shank','FR_shank','BL_shank','BR_shank']
        sensor=self._env.scene['contact_forces'];ids,_=sensor.find_bodies(bodies,preserve_order=True)
        contact.effort=torch.linalg.vector_norm(sensor.data.net_forces_w[0,ids],dim=-1).tolist()
        self._publish_input(self._contact_pub,contact)
