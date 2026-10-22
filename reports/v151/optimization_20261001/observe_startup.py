"""Read-only observer: subscribe only; never publish sensor/control messages."""
import json,time
import rclpy
from sensor_msgs.msg import JointState,Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from v14_ros2_control import contract
rclpy.init();node=rclpy.create_node('v151_readonly_startup_observer',namespace='/v151')
seen=set();subs=[]
def receive(topic,msg):
    if topic in seen:return
    seen.add(topic)
    value=json.loads(msg.data) if topic=='status' else {'stamp':contract.stamp_ns(msg.header.stamp),'frame':msg.header.frame_id}
    if topic=='status':value={k:value.get(k) for k in ('stamp_ns','mode','support_feet')}
    print(json.dumps({'topic':topic,'first':value}),flush=True)
for typ,topic in [(JointState,'joint_states'),(Odometry,'odom'),(Image,'depth/image'),(JointState,'foot_contacts'),(JointState,'joint_command'),(String,'status')]:
    subs.append(node.create_subscription(typ,topic,lambda m,k=topic:receive(k,m),contract.lockstep_qos()))
deadline=time.monotonic()+180
try:
    while time.monotonic()<deadline and len(seen)<6:rclpy.spin_once(node,timeout_sec=.1)
    print(json.dumps({'seen':sorted(seen),'publisher_counts':[s.get_publisher_count() for s in subs]}),flush=True)
finally:node.destroy_node();rclpy.shutdown()
