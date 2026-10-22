"""Instantiate the extracted ROS2 runtime with the opt-in deeper crouch."""
import json,sys
from pathlib import Path
import rclpy
from v15_hardware_bridge.cad_controller import HardwareController
root=Path(sys.argv[1]).resolve()
rclpy.init(args=['--ros-args','-p','actor_checkpoint:='+str(root/'checkpoints/model_15699.pt'),
 '-p','student_checkpoint:='+str(root/'checkpoints/v12_student_best.pt'),
 '-p','model_path:='+str(root/'model/quadruped_cad.xml'),
 '-p','initial_mode:=STAND','-p','sit_style:=deep_crouch','-p','height_drop:=0.18',
 '-p','posture_transition_s:=2.0','-p','stand_forward_shift_m:=0.0','-p','crouch_forward_shift_m:=0.0'])
node=HardwareController()
try:
 assert Path(sys.modules[type(node).__module__].__file__).resolve().is_relative_to(root)
 assert node.initial=='STAND' and node.sit_style=='deep_crouch' and node.drop==.18
 assert node.supervisor.rise_s==2. and node.demo is False
 node.supervisor.request('sit'); assert node.supervisor.requested=='sit'
 imported=sorted(n for n in sys.modules if n.split('.')[0] in {'isaaclab','isaacsim','omni','pxr'})
 assert not imported,imported
 print(json.dumps(dict(passed=True,initial_mode=node.initial,sit_style=node.sit_style,
  height_drop_m=node.drop,posture_transition_s=node.supervisor.rise_s,simulator_imports=imported,
  scope='Extracted x86 CPU/Jazzy runtime initialization; not hardware timing or physical testing'),indent=2))
finally:
 node.destroy_node();rclpy.shutdown()
