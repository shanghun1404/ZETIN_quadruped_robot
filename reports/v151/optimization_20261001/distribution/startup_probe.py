"""Execute the archived startup method with deliberately unmatched endpoints.

This demonstrates an incomplete readiness predicate, not job616 root cause.
No ROS/Isaac dependencies or production edits required.
"""
import ast
import json
import sys
import time
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
source=ROOT/'reports/v151/integration/616/source/config/v151_ros2_bridge.py'
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='V151Ros2Action')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_start_ros')
method.decorator_list=[]

class Endpoint:
    def __init__(self,topic): self.topic=topic
    def get_subscription_count(self): return 0 if self.topic=='foot_contacts' else 1
    def get_publisher_count(self): return 0 if self.topic=='status' else 1

class Node:
    def __init__(self): self.endpoints={}
    def create_publisher(self,typ,topic,qos):
        self.endpoints[topic]=Endpoint(topic); return self.endpoints[topic]
    def create_subscription(self,typ,topic,callback,qos):
        self.endpoints[topic]=Endpoint(topic); return self.endpoints[topic]

node=Node(); spins=[]
rclpy=types.ModuleType('rclpy')
rclpy.ok=lambda:True
rclpy.create_node=lambda *a,**kw:node
rclpy.spin_once=lambda *a,**kw:spins.append(1)
sys.modules['rclpy']=rclpy
for package,names in [('nav_msgs',['Odometry']),('rosgraph_msgs',['Clock']),('sensor_msgs',['Image','JointState']),('std_msgs',['String'])]:
    module=types.ModuleType(package+'.msg')
    for name in names: setattr(module,name,type(name,(),{}))
    sys.modules[package]=types.ModuleType(package)
    sys.modules[package+'.msg']=module
contract=types.SimpleNamespace(lockstep_qos=lambda:None,JOINT_STATES='joint_states',ODOM='odom',DEPTH_IMAGE='depth/image',JOINT_COMMAND='joint_command')
ns={'time':time,'contract':contract}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
adapter=types.SimpleNamespace(_on_command=lambda _:None,_on_status=lambda _:None)
ns['_start_ros'](adapter,types.SimpleNamespace(namespace='/v151',startup_timeout_s=.01))
result=dict(startup_returned=True,spin_count=len(spins),contact_subscribers=node.endpoints['foot_contacts'].get_subscription_count(),status_publishers=node.endpoints['status'].get_publisher_count(),
    inference='Archived readiness predicate permits return while mandatory contact/status endpoints have zero peers. TRANSIENT_LOCAL durability means this is not proof of lost first messages or root cause of616.')
assert result['startup_returned'] and result['spin_count']==0
assert result['contact_subscribers']==result['status_publishers']==0
out=Path(__file__).with_name('startup_probe.json')
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
