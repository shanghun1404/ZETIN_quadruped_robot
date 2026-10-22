"""Native Jazzy test: real ROS messages through a real PTY serial device."""
import importlib.util
import os
import pty
import select
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

if importlib.util.find_spec('rclpy') is None:
    pytest.skip('requires native ROS2 Jazzy; run on ws1',allow_module_level=True)

sys.path.insert(0,str(Path(__file__).parents[1]))
import rclpy
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from v14_ros2_control import contract
from v15_hardware_bridge.protocol import (Frame,StreamParser,encode_frame,pack_feedback,
    unpack_command,HELLO,HELLO_ACK,FEEDBACK,COMMAND)


def test_gateway_atomic_command_and_feedback_over_pty(tmp_path):
    import json
    target=Path(__file__).parents[1]/'v15_hardware_bridge/usb_gateway.py'
    assert target.exists(),'ROS2 USB gateway is missing'
    from v15_hardware_bridge.usb_gateway import UsbGateway
    master,slave=pty.openpty();path=os.ttyname(slave)
    cfg=tmp_path/'hardware.json';cfg.write_text(json.dumps(dict(serial_port=path,baud=921600)))
    stop=threading.Event();commands=[];errors=[]
    def board():
        parser=StreamParser();session=None;sequence=0;stamp=0;deadline=time.monotonic()
        try:
            while not stop.is_set():
                ready,_,_=select.select([master],[],[],.001)
                if ready:
                    for frame in parser.feed(os.read(master,4096)):
                        if frame.kind==HELLO:
                            session=frame.session
                            os.write(master,encode_frame(Frame(HELLO_ACK,session,0,0,struct.pack('<I',1))))
                        elif frame.kind==COMMAND:commands.append(unpack_command(frame.payload))
                if session is not None and time.monotonic()>=deadline:
                    payload=pack_feedback([.1]*12,[0.]*12,[0.]*12)
                    os.write(master,encode_frame(Frame(FEEDBACK,session,sequence,stamp,payload)))
                    sequence+=1;stamp+=5_000_000;deadline=time.monotonic()+.005
        except Exception as error:errors.append(str(error))
    thread=threading.Thread(target=board);thread.start()
    rclpy.init()
    gateway=UsbGateway(parameter_overrides=[Parameter('hardware_config',value=str(cfg))])
    operator=rclpy.create_node('usb_test_operator',namespace='/v15real')
    qpub=operator.create_publisher(JointState,'joint_command',contract.lockstep_qos())
    spub=operator.create_publisher(String,'status',contract.lockstep_qos())
    feedback=[];operator.create_subscription(JointState,'raw/joint_states',feedback.append,10)
    executor=SingleThreadedExecutor();executor.add_node(gateway);executor.add_node(operator)
    try:
        deadline=time.monotonic()+3.
        while (not feedback or qpub.get_subscription_count()==0 or spub.get_subscription_count()==0) and time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.001)
        assert feedback
        msg=JointState();msg.name=list(contract.JOINT_NAMES);msg.position=[i*.1 for i in range(12)];msg.effort=[0.]*12
        qpub.publish(msg)
        for _ in range(10):executor.spin_once(timeout_sec=.001)
        assert not commands,'position without matching gains must not be sent'
        status=String();status.data=json.dumps(dict(stamp_ns=0,command_schema=2,stiffness_nm_rad=150.,damping_nms_rad=4.))
        spub.publish(status)
        deadline=time.monotonic()+1.
        while not commands and time.monotonic()<deadline:executor.spin_once(timeout_sec=.001)
        assert commands and not errors
        assert commands[0]['enabled'] is False
        assert commands[0]['position']==pytest.approx([0.,.2,.8,.1,.3,.9,.5,.7,1.1,.4,.6,1.])
        assert commands[0]['kp']==pytest.approx([150.]*12) and commands[0]['kd']==pytest.approx([4.]*12)
        assert feedback[0].position==pytest.approx([.1]*12)
        assert set(feedback[0].name)==set(contract.JOINT_NAMES)
    finally:
        gateway.destroy_node();operator.destroy_node();executor.shutdown();rclpy.shutdown()
        stop.set();thread.join(timeout=2.);os.close(master);os.close(slave)


def test_gateway_accepts_periodic_native_mcu_feedback(tmp_path):
    """Use the real C endpoint rather than the Python protocol responder."""
    import json
    import subprocess
    from v15_hardware_bridge.usb_gateway import UsbGateway
    firmware=next((parent/'firmware/stm32g474_usb'
        for parent in Path(__file__).resolve().parents
        if (parent/'firmware/stm32g474_usb/Makefile').is_file()),None)
    assert firmware is not None,'the native integration test requires the firmware sources'
    executable=firmware/'build/qd_emulator'
    assert executable.is_file(),'build the native MCU emulator before the Jazzy integration test'
    measured=tmp_path/'measured.bin'
    measured.write_bytes(pack_feedback([.1]*12,[.2]*12,[.3]*12))
    master,slave=pty.openpty()
    process=subprocess.Popen([str(executable),'--measurement',str(measured)],
        stdin=master,stdout=master,stderr=subprocess.PIPE)
    cfg=tmp_path/'hardware.json'
    cfg.write_text(json.dumps(dict(serial_port=os.ttyname(slave),baud=921600)))
    rclpy.init()
    gateway=UsbGateway(parameter_overrides=[Parameter('hardware_config',value=str(cfg))])
    observer=rclpy.create_node('native_mcu_observer',namespace='/v15real')
    states=[]
    observer.create_subscription(JointState,'raw/joint_states',states.append,10)
    executor=SingleThreadedExecutor();executor.add_node(gateway);executor.add_node(observer)
    try:
        deadline=time.monotonic()+2.
        while len(states)<5 and not gateway.failed and time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.001)
        assert not gateway.failed,'real C feedback must satisfy the gateway 5 ms clock contract'
        assert gateway.session.acknowledged and len(states)>=5
        stamps=[contract.stamp_ns(state.header.stamp) for state in states]
        assert all(b-a==5_000_000 for a,b in zip(stamps,stamps[1:]))
        assert states[-1].position==pytest.approx([.1]*12)
        assert states[-1].velocity==pytest.approx([.2]*12)
        assert states[-1].effort==pytest.approx([.3]*12)
    finally:
        gateway.destroy_node();observer.destroy_node();executor.shutdown();rclpy.shutdown()
        process.terminate();process.communicate(timeout=2.)
        os.close(master);os.close(slave)
