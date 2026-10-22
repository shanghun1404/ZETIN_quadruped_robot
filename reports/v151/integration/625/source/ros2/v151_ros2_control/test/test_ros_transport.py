"""Actual DDS integration with a synthetic sensor publisher (no simulated physics)."""
import json
import os
import time
import array
from unittest.mock import patch
import numpy as np
import pytest


def retry_adapter(monkeypatch):
    """Run real bridge transport methods without loading Isaac's application."""
    import ast
    import types
    import sys
    from pathlib import Path
    source=Path(__file__).resolve().parents[3]/'config/v151_ros2_bridge.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='V151Ros2Action')
    clock=types.SimpleNamespace(now=0.)
    contract=types.SimpleNamespace(PHYSICS_DT_NS=5_000_000,stamp_ns=lambda s:s.sec*1_000_000_000+s.nanosec,
        step_stamp=lambda step:divmod(step*5_000_000,1_000_000_000),
        ordered=lambda names,values:list(values))
    ns={'V14Ros2JointTargetAction':object,'time':types.SimpleNamespace(monotonic=lambda:clock.now),'json':json,'contract':contract}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
    adapter=ns['V151Ros2Action']()
    adapter._commands={}; adapter._statuses={}; adapter._pending_inputs=[]
    adapter._physics_step=1; adapter.retry_count=0
    adapter._node=object(); adapter.cfg=types.SimpleNamespace(step_timeout_s=.6)
    adapter._state_pub=adapter._odom_pub=adapter._image_pub=adapter._contact_pub=types.SimpleNamespace(get_subscription_count=lambda:1)
    def spin_once(*args,**kwargs): clock.now+=.05
    monkeypatch.setitem(sys.modules,'rclpy',types.SimpleNamespace(spin_once=spin_once))
    return adapter,clock


def test_bridge_retry_reuses_identical_inputs_until_both_replies_arrive(monkeypatch):
    import types
    adapter,clock=retry_adapter(monkeypatch)
    sent=[]; messages=[object() for _ in range(4)]
    def publish(message):
        sent.append((clock.now,message))
        if message is messages[-1] and len(sent)==8: adapter._commands[5_000_000]=('command',)
        if message is messages[-1] and len(sent)==12: adapter._statuses[5_000_000]={'stamp_ns':5_000_000}
    pub=types.SimpleNamespace(publish=publish)
    for message in messages: adapter._publish_input(pub,message)
    adapter._wait_for_reply(5_000_000)
    assert adapter.retry_count==2 and adapter._physics_step==1
    assert len(sent)==12
    for index,(_,message) in enumerate(sent): assert message is messages[index%4]
    assert sent[4][0]>=.25 and sent[8][0]>=.5
    assert len(adapter._pending_inputs)==4


def test_bridge_retry_preserves_timeout_and_ignores_stale_replies(monkeypatch):
    import types
    adapter,clock=retry_adapter(monkeypatch)
    with pytest.raises(TimeoutError,match='command_received=False, status_received=False'):
        adapter._wait_for_reply(5_000_000)
    assert .6<=clock.now<=.7 and adapter._physics_step==1
    message=types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=0,nanosec=0)),name=[],position=[],effort=[])
    adapter._on_command(message); adapter._on_status(types.SimpleNamespace(data='{"stamp_ns":0}'))
    assert not adapter._commands and not adapter._statuses

@pytest.mark.skipif(not os.environ.get('V151_ACTOR'), reason='requires explicit checkpoint/runtime paths')
@pytest.mark.parametrize('initial_mode',['SIT','STAND'])
@pytest.mark.parametrize('crouch_shift',[0.,.12])
def test_sensor_stamp_drives_one_command_and_exposes_wbc_diagnostics(initial_mode,crouch_shift):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState, Image
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from v14_ros2_control import contract
    from v151_ros2_control.controller_node import ControllerNode
    rclpy.init(args=['--ros-args','-p','actor_checkpoint:='+os.environ['V151_ACTOR'],
        '-p','student_checkpoint:='+os.environ['V151_STUDENT'],'-p','model_path:='+os.environ['V151_MODEL'],
        '-p','initial_mode:='+initial_mode,'-p','crouch_forward_shift_m:='+str(crouch_shift)])
    ctl=ControllerNode();tester=rclpy.create_node('v151_test_sensors')
    executor=SingleThreadedExecutor();executor.add_node(ctl);executor.add_node(tester)
    qos=contract.lockstep_qos();received=[];statuses=[]
    pubs=[tester.create_publisher(typ,topic,qos) for typ,topic in
        [(JointState,'joint_states'),(Odometry,'odom'),(Image,'depth/image'),(JointState,'foot_contacts')]]
    tester.create_subscription(JointState,'joint_command',received.append,qos)
    tester.create_subscription(String,'status',statuses.append,qos)
    try:
        deadline=time.monotonic()+10
        while any(p.get_subscription_count()<1 for p in pubs):
            executor.spin_once(timeout_sec=.01)
            assert time.monotonic()<deadline
        state=JointState();state.name=list(contract.JOINT_NAMES)
        state.position=[contract.joint_default(n) for n in contract.JOINT_NAMES];state.velocity=[0.]*12
        odom=Odometry();odom.pose.pose.position.z=.464;odom.pose.pose.orientation.w=1.
        image=Image();image.height=84;image.width=84;image.encoding='32FC1';image.step=84*4
        image.data=array.array('B',np.full((84,84),2.,dtype='<f4').tobytes())
        contacts=JointState();contacts.name=['FL_foot','FR_foot','RL_foot','RR_foot'];contacts.effort=[111.]*4
        # A retry while the image is still missing must not process the stamp.
        for index in (0,1,3,0): pubs[index].publish([state,odom,image,contacts][index])
        for _ in range(12): executor.spin_once(timeout_sec=.005)
        assert not received and not statuses
        pubs[2].publish(image)
        while not received or not statuses:
            executor.spin_once(timeout_sec=.01)
            assert time.monotonic()<deadline
        assert len(received)==1
        assert tuple(received[0].name)==contract.JOINT_NAMES
        assert np.isfinite(received[0].position).all()
        status=json.loads(statuses[0].data)
        assert status['support_feet']==4
        assert status['crouch_forward_shift_m']==crouch_shift
        assert status['wbc']['finite']
        assert status['wbc']['feasible']
        if initial_mode=='STAND':
            assert np.max(np.abs(ctl.crouch-ctl.nominal))>.2
            low=ctl.standing.contact_positions(ctl.crouch,[0.,0.,ctl.stand_height-ctl.drop],[1.,0.,0.,0.])
            high=ctl.standing.contact_positions(ctl.nominal,[0.,0.,ctl.stand_height],[1.,0.,0.,0.])
            np.testing.assert_allclose(low,high,atol=2e-6)
        from copy import deepcopy
        # Simulate lost outputs by requesting the same completed stamp again.
        # Any accidental rerun of a stateful controller stage fails the test.
        def replay_without_computing():
            expected_command=deepcopy(received[-1]); expected_status=statuses[-1].data
            command_count,status_count=len(received),len(statuses)
            last=ctl.last; warm=ctl.standing._warm; output=ctl.output.copy()
            with patch.object(ctl.policy,'step',side_effect=AssertionError('policy rerun')), \
                 patch.object(ctl.standing,'compute',side_effect=AssertionError('WBC rerun')), \
                 patch.object(ctl.filter,'step',side_effect=AssertionError('filter rerun')), \
                 patch.object(ctl.supervisor,'step',side_effect=AssertionError('supervisor rerun')):
                pubs[0].publish(state)
                replay_deadline=time.monotonic()+2
                while len(received)==command_count or len(statuses)==status_count:
                    executor.spin_once(timeout_sec=.01)
                    assert time.monotonic()<replay_deadline, 'duplicate did not replay cached outputs'
            # Compare all ROS fields; CDR alignment padding is not message data.
            assert received[-1]==expected_command
            assert statuses[-1].data==expected_status
            assert ctl.last==last and ctl.standing._warm is warm and ctl.output==output
        replay_without_computing()
        # The observed failure was on a 200 Hz step without policy inputs.
        state.header.stamp.nanosec=5_000_000
        before=len(received)
        pubs[0].publish(state)
        while len(received)==before or json.loads(statuses[-1].data)['stamp_ns']!=5_000_000:
            executor.spin_once(timeout_sec=.01)
            assert time.monotonic()<deadline
        replay_without_computing()
        # Older stamps must not replay an obsolete actuator command.
        counts=len(received),len(statuses)
        state.header.stamp.nanosec=0
        pubs[0].publish(state)
        for _ in range(12): executor.spin_once(timeout_sec=.005)
        assert (len(received),len(statuses))==counts
    finally:
        executor.shutdown();ctl.destroy_node();tester.destroy_node();rclpy.shutdown()


@pytest.mark.skipif(not os.environ.get('V151_ACTOR'), reason='requires explicit checkpoint/runtime paths')
def test_after_walk_only_preserves_baseline_before_support_latch():
    """A nonzero configured shift must not alter the original prewalk path."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState,Image
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from v14_ros2_control import contract
    from v151_ros2_control.controller_node import ControllerNode

    def run_case(shift,namespace):
        rclpy.init(args=['--ros-args','-r','__ns:='+namespace,
            '-p','actor_checkpoint:='+os.environ['V151_ACTOR'],
            '-p','student_checkpoint:='+os.environ['V151_STUDENT'],
            '-p','model_path:='+os.environ['V151_MODEL'],'-p','initial_mode:=SIT',
            '-p','demo:=true','-p','stand_forward_shift_m:=0.03',
            '-p','crouch_forward_shift_m:='+str(shift),'-p','crouch_shift_after_walk_only:=true'])
        ctl=ControllerNode(); tester=rclpy.create_node('scope_test_sensors')
        executor=SingleThreadedExecutor(); executor.add_node(ctl); executor.add_node(tester)
        qos=contract.lockstep_qos(); received=[]; statuses=[]
        pubs=[tester.create_publisher(typ,topic,qos) for typ,topic in
            [(JointState,'joint_states'),(Odometry,'odom'),(Image,'depth/image'),(JointState,'foot_contacts')]]
        tester.create_subscription(JointState,'joint_command',received.append,qos)
        tester.create_subscription(String,'status',statuses.append,qos)
        records=[]
        try:
            deadline=time.monotonic()+10
            while any(p.get_subscription_count()<1 for p in pubs):
                executor.spin_once(timeout_sec=.01)
                assert time.monotonic()<deadline
            q,height=ctl.standing.crouch_pose(.10)
            state=JointState(); state.name=list(contract.JOINT_NAMES)
            state.position=q[ctl.to_policy].tolist(); state.velocity=[0.]*12
            odom=Odometry(); odom.pose.pose.position.z=height; odom.pose.pose.orientation.w=1.
            image=Image(); image.height=image.width=84; image.encoding='32FC1'; image.step=336
            image.data=array.array('B',np.full((84,84),2.,dtype='<f4').tobytes())
            contacts=JointState(); contacts.name=['FL_foot','FR_foot','RL_foot','RR_foot']; contacts.effort=[111.]*4
            for t in (0.,.50,1.,3.50,6.,6.02):
                if t==6.02:
                    # Isolate the existing latch's consumer: the real handover
                    # sets this field after initializing the standing anchors.
                    ctl.measured_support=True
                    ctl.demo=False; ctl.supervisor.mode='SIT'; ctl.supervisor.request('sit')
                count=len(received); count_status=len(statuses)
                sec,nanosec=divmod(round(t*1e9),1_000_000_000)
                for pub,msg in zip(pubs,[state,odom,image,contacts]):
                    msg.header.stamp.sec=sec; msg.header.stamp.nanosec=nanosec; pub.publish(msg)
                while len(received)==count or len(statuses)==count_status:
                    executor.spin_once(timeout_sec=.01)
                    assert time.monotonic()<deadline
                status=json.loads(statuses[-1].data)
                assert status['crouch_shift_after_walk_only'] is True
                records.append((np.array(received[-1].effort),status['wbc_vectors']))
            return records
        finally:
            executor.shutdown(); ctl.destroy_node(); tester.destroy_node(); rclpy.shutdown()

    baseline=run_case(0.,'/scope_baseline')
    candidate=run_case(.12,'/scope_candidate')
    for original,optimized in zip(baseline[:-1],candidate[:-1]):
        np.testing.assert_array_equal(original[0],optimized[0])
        assert original[1]==optimized[1]
    before=np.array(baseline[-1][1]['base_position_reference_world'])
    after=np.array(candidate[-1][1]['base_position_reference_world'])
    np.testing.assert_allclose(after-before,[0.,.12,0.],atol=1e-12)
