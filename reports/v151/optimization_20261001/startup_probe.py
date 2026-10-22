"""Two-process real DDS first-command probe; no physics or robot actuation."""
import array
import json
import os
from pathlib import Path
import subprocess
import sys
import time
kit=None
if '--kit' in sys.argv:
    from isaacsim import SimulationApp
    kit=SimulationApp({'headless':True})
import numpy as np
import rclpy
from sensor_msgs.msg import JointState,Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from v14_ros2_control import contract
from v151_ros2_control.motion import CANONICAL

root=Path('/home/shanghun1404/v151_ros2')
initial=json.loads((root/'model/crouch.json').read_text())
log=Path(sys.argv[1]);handle=log.open('w')
proc=subprocess.Popen([sys.executable,'-m','v151_ros2_control.controller_node','--ros-args','-r','__ns:=/v151',
    '-p','actor_checkpoint:='+os.environ['V151_ACTOR'],'-p','student_checkpoint:='+os.environ['V151_STUDENT'],
    '-p','model_path:='+os.environ['V151_MODEL'],'-p','initial_mode:=SIT',
    '-p','stand_forward_shift_m:=0.09','-p','crouch_forward_shift_m:=0.15'],stdout=handle,stderr=subprocess.STDOUT)
rclpy.init();node=rclpy.create_node('v151_first_input_probe',namespace='/v151')
qos=contract.lockstep_qos();received={}
pubs=[node.create_publisher(typ,topic,qos) for typ,topic in
      [(JointState,'joint_states'),(Odometry,'odom'),(Image,'depth/image'),(JointState,'foot_contacts')]]
subs=[node.create_subscription(typ,topic,lambda m,k=topic:received.__setitem__(k,m),qos)
      for typ,topic in [(JointState,'joint_command'),(String,'status')]]
try:
    deadline=time.monotonic()+20
    while not(all(p.get_subscription_count()>=1 for p in pubs) and all(s.get_publisher_count()>=1 for s in subs)):
        rclpy.spin_once(node,timeout_sec=.01)
        if proc.poll() is not None: raise RuntimeError('Controller exited: '+log.read_text())
        if time.monotonic()>deadline: raise TimeoutError('Endpoint discovery failed')
    state=JointState();state.name=list(contract.JOINT_NAMES)
    state.position=[initial['joint_position'][CANONICAL.index(n)] for n in state.name];state.velocity=[0.]*12
    odom=Odometry();odom.pose.pose.position.z=initial['height'];odom.pose.pose.orientation.w=1.
    depth=Image();depth.height=84;depth.width=84;depth.encoding='32FC1';depth.step=84*4
    depth.data=array.array('B',np.full((84,84),2.,dtype='<f4').tobytes())
    contact=JointState();contact.name=[leg+'_foot' for leg in ('FL','FR','RL','RR')];contact.effort=[0.]*4
    # Match adapter order and actual initial zero contacts. Publish each only once.
    start=time.monotonic()
    for index,msg in [(1,odom),(2,depth),(3,contact),(0,state)]: pubs[index].publish(msg)
    while len(received)<2:
        rclpy.spin_once(node,timeout_sec=.001)
        if proc.poll() is not None: raise RuntimeError('Controller exited: '+log.read_text())
        if time.monotonic()-start>10: raise TimeoutError('Missing command/status: '+log.read_text())
    status=json.loads(received['status'].data)
    assert status['stamp_ns']==0 and status['mode']=='SIT'
    assert np.isfinite(received['joint_command'].effort).all()
    print(json.dumps(dict(first_command_received=True,elapsed_s=time.monotonic()-start,
        domain=os.environ['ROS_DOMAIN_ID'],crouch_shift=status['crouch_forward_shift_m'],wbc=status['wbc'])))
finally:
    node.destroy_node();rclpy.shutdown();proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill();proc.wait()
    handle.close()
if kit is not None:
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
