"""Native ROS contracts for the separate planner and measured-joint command node."""
import array
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.parameter import Parameter
from std_msgs.msg import String
from v14_ros2_control import contract
from v151_ros2_control.motion import CANONICAL


def bundle():
    return next((p for p in Path(__file__).resolve().parents if (p/'model/quadruped_cad.xml').is_file()),None)


def test_planner_publishes_only_one_atomic_plan_per_policy_tick():
    root=bundle()
    if root is None:pytest.skip('requires packaged CAD model and checkpoints')
    from v15_hardware_bridge.cad_controller import HardwareController
    rclpy.init(args=['--ros-args','-r','__ns:=/planner_contract','-p','planner_only:=true',
        '-p','initial_mode:=STAND','-p','wbc_backend:=cpp','-p','device:=cpu',
        '-p','actor_checkpoint:='+str(root/'checkpoints/model_15699.pt'),
        '-p','student_checkpoint:='+str(root/'checkpoints/v12_student_best.pt'),
        '-p','model_path:='+str(root/'model/quadruped_cad.xml')])
    node=observer=executor=None
    try:
        node=HardwareController()
        assert getattr(node,'planner_only',False),'50 Hz planner mode is missing'
        observer=rclpy.create_node('planner_observer',namespace='/planner_contract')
        executor=SingleThreadedExecutor();executor.add_node(node);executor.add_node(observer)
        plans=[];commands=[]
        observer.create_subscription(String,'control_plan',lambda m:plans.append(json.loads(m.data)),contract.lockstep_qos())
        observer.create_subscription(JointState,'joint_command',commands.append,contract.lockstep_qos())
        q=node.standing.nominal_joint_position[[CANONICAL.index(n) for n in contract.JOINT_NAMES]]
        state=JointState(name=list(contract.JOINT_NAMES),position=q.tolist(),velocity=[0.]*12)
        odom=Odometry();odom.pose.pose.orientation.w=1.;odom.pose.pose.position.z=node.standing.nominal_height
        image=Image(height=84,width=84,encoding='32FC1',step=336,
                    data=array.array('B',np.full((84,84),2.,dtype='<f4').tobytes()))
        contacts=JointState(effort=[50.]*4)
        # A partial sample from startup must not block a later complete set.
        node.on_state(state)
        for step in range(4,12):
            state.header.stamp.sec,state.header.stamp.nanosec=contract.step_stamp(step)
            if step%4==0:
                for message,buffer in ((odom,node.odoms),(image,node.images),(contacts,node.contacts)):
                    message.header.stamp=state.header.stamp;node.add(buffer,message)
            node.on_state(state)
        deadline=time.monotonic()+3.
        while len(plans)<2 and time.monotonic()<deadline:executor.spin_once(timeout_sec=.01)
        assert [p['source_stamp_ns'] for p in plans]==[20_000_000,40_000_000]
        assert all(len(p['raw_target'])==len(p['standing_torque'])==12 for p in plans)
        assert all(p['plan_schema']==1 and p['status']['mode']=='STAND' for p in plans)
        assert commands==[]
    finally:
        if node is not None:node.destroy_node()
        if observer is not None:observer.destroy_node()
        if executor is not None:executor.shutdown()
        if rclpy.ok():rclpy.shutdown()


def test_command_process_latches_fault_when_planner_stops():
    assert importlib.util.find_spec('v15_hardware_bridge.command_node') is not None,'Separate command node is missing'
    from v15_hardware_bridge.command_node import JointCommandNode
    rclpy.init();node=observer=executor=None
    try:
        node=JointCommandNode(namespace='/command_fault_contract')
        observer=rclpy.create_node('command_fault_observer',namespace='/command_fault_contract')
        executor=SingleThreadedExecutor();executor.add_node(node);executor.add_node(observer)
        commands=[];faults=[]
        observer.create_subscription(JointState,'joint_command',commands.append,contract.lockstep_qos())
        observer.create_subscription(String,'hardware_fault',faults.append,10)
        q=[contract.joint_default(n) for n in contract.JOINT_NAMES]
        packet=dict(plan_schema=1,source_stamp_ns=0,generation=0,raw_target=q,standing_torque=[0.]*12,
                    status=dict(mode='STAND',walk_weight=0.,height_fraction=1.,command=[0.,0.,0.]))
        node.on_plan(String(data=json.dumps(packet)))
        state=JointState(name=list(contract.JOINT_NAMES),position=q,velocity=[0.]*12)
        node.on_state(state)
        deadline=time.monotonic()+3.
        while (not commands or not faults) and time.monotonic()<deadline:executor.spin_once(timeout_sec=.005)
        assert len(commands)==1 and faults and node.failed
        packet['source_stamp_ns']=20_000_000
        node.on_plan(String(data=json.dumps(packet)))
        state.header.stamp.nanosec=20_000_000;node.on_state(state)
        for _ in range(5):executor.spin_once(timeout_sec=.005)
        assert len(commands)==1 and node.failed
    finally:
        if node is not None:node.destroy_node()
        if observer is not None:observer.destroy_node()
        if executor is not None:executor.shutdown()
        if rclpy.ok():rclpy.shutdown()


def test_plan_expiring_during_command_computation_is_not_published(monkeypatch):
    from types import SimpleNamespace
    from v15_hardware_bridge import command_node
    rclpy.init();node=None
    try:
        node=command_node.JointCommandNode(namespace='/command_expiry_contract')
        q=[contract.joint_default(n) for n in contract.JOINT_NAMES]
        node.stage.update(dict(plan_schema=1,source_stamp_ns=0,generation=0,
            raw_target=q,standing_torque=[0.]*12,status=dict(mode='STAND',
            walk_weight=0.,height_fraction=1.,command=[0.,0.,0.])),0)
        published=[];node.command_pub=SimpleNamespace(publish=published.append)
        clock=iter((0,60_000_000))
        monkeypatch.setattr(command_node.time,'monotonic_ns',lambda:next(clock))
        node.on_state(JointState(name=list(contract.JOINT_NAMES),position=q,velocity=[0.]*12))
        assert node.failed
        assert published==[]
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_overlapped_planner_preserves_sequential_blend_outputs(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('requires CUDA runtime')
    root=bundle()
    if root is None:pytest.skip('requires packaged CAD model and checkpoints')
    from v15_hardware_bridge.cad_controller import HardwareController
    rclpy.init();nodes=[]
    try:
        settings=dict(initial_mode='STAND',wbc_backend='cpp',device=device,
            actor_checkpoint=str(root/'checkpoints/model_15699.pt'),
            student_checkpoint=str(root/'checkpoints/v12_student_best.pt'),
            model_path=str(root/'model/quadruped_cad.xml'))
        for planning in (True,False):
            nodes.append(HardwareController(namespace='/parallel' if planning else '/sequential',
                parameter_overrides=[Parameter(k,value=v) for k,v in dict(settings,planner_only=planning).items()]))
        planner,legacy=nodes
        assert planner.policy.worker_pid!=os.getpid(), 'Policy and WBC need separate Python interpreters'
        request=Twist();request.linear.y=.4
        for node in nodes:
            # Exercise inference/WBC overlap immediately; measured readiness
            # and default settle timing are covered by supervisor tests.
            node.supervisor.settle_s=0.
            node.command(request)
        q=planner.standing.nominal_joint_position[[CANONICAL.index(n) for n in contract.JOINT_NAMES]]
        standing_models=[node.standing for node in nodes]
        for step in range(20):
            if step==12:
                for node in nodes:
                    node.supervisor.mode='BRAKE';node.supervisor.entered=-1.
                    node.supervisor.request('stand')
            for node in nodes:
                state=JointState(name=list(contract.JOINT_NAMES),position=q.tolist(),velocity=[0.]*12)
                state.header.stamp.sec,state.header.stamp.nanosec=contract.step_stamp(step)
                if step%4==0:
                    odom=Odometry();odom.pose.pose.orientation.w=1.
                    odom.pose.pose.position.z=node.standing.nominal_height
                    image=Image(height=84,width=84,encoding='32FC1',step=336,
                        data=array.array('B',np.full((84,84),2.,dtype='<f4').tobytes()))
                    contacts=JointState(effort=[50.]*4)
                    for message,buffer in ((odom,node.odoms),(image,node.images),(contacts,node.contacts)):
                        message.header.stamp=state.header.stamp;node.add(buffer,message)
                node.on_state(state)
            if step%4==0:
                packet=json.loads(planner._last_plan.data)
                np.testing.assert_allclose(packet['raw_target'],legacy.filter.raw.numpy(),atol=1e-7)
                np.testing.assert_allclose(packet['standing_torque'],legacy.stand_tau,atol=1e-10)
                np.testing.assert_allclose(planner.policy.last_action.numpy(),legacy.policy.last_action.numpy(),atol=1e-7)
                if step==8:assert packet['status']['mode']=='BLEND_WALK' and packet['status']['walk_weight']>0.
                if step>=12:assert all(node.standing is model for node,model in zip(nodes,standing_models))
        assert packet['status']['mode']=='BLEND_STAND' and packet['status']['walk_weight']<1.
        sample=dict(joint_pos=q,joint_vel=np.zeros(12),lin_vel_b=np.zeros(3),ang_vel_b=np.zeros(3),
            quat_wxyz=[1.,0.,0.,0.],command=[0.,.4,0.],depth_m=np.full((84,84),2.,dtype=np.float32))
        before=planner.policy.last_action.clone()
        with pytest.raises(ValueError,match='depth'):
            planner.policy.step(**dict(sample,depth_m=np.zeros((2,2))))
        torch.testing.assert_close(planner.policy.last_action,before)
        planner.policy.reset();legacy.policy.reset()
        actual=planner.policy.step(**sample);expected=legacy.policy.step(**sample)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-7)
    finally:
        for node in nodes:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
