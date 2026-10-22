"""ROS2 motor command/status pairing to the STM32 USB protocol; no Isaac imports."""
import json
import secrets
import struct
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import serial
from v14_ros2_control import contract
from .core import CommandMatcher,SessionGuard,validate_hardware_config
from .serial_transport import write_frame_once
from .protocol import (Frame,StreamParser,encode_frame,pack_command,unpack_feedback,
                       JOINT_NAMES,HELLO,HELLO_ACK,COMMAND,FEEDBACK,STOP)


class UsbGateway(Node):
    def __init__(self,**kwargs):
        kwargs.setdefault('namespace','/v15real')
        super().__init__('v15_usb_gateway',**kwargs)
        param=lambda name,default:self.declare_parameter(name,default).value
        config=json.loads(Path(param('hardware_config','')).read_text())
        self.enabled=bool(param('enable_motors',False))
        validate_hardware_config(config,self.enabled)
        port=param('serial_port',config.get('serial_port','/dev/ttyACM0'))
        self.serial=serial.Serial(port,baudrate=int(config.get('baud',921600)),timeout=0,write_timeout=0,exclusive=True)
        self.matcher=CommandMatcher();self.parser=StreamParser();self.session=SessionGuard(secrets.randbelow(0xffffffff)+1)
        self.sequence=0;self.started=time.monotonic_ns();self.base_stamp=None;self.failed=False
        self.feedback_pub=self.create_publisher(JointState,'raw/joint_states',10)
        self.diag_pub=self.create_publisher(String,'hardware_status',10)
        self.create_subscription(JointState,'joint_command',self.on_command,contract.lockstep_qos())
        self.create_subscription(String,'status',self.on_status,contract.lockstep_qos())
        self.create_subscription(String,'hardware_fault',lambda message:self.fail(message.data),10)
        self._write(HELLO,0,b'')
        self.timer=self.create_timer(.001,self.pump)
        self.get_logger().info(f'USB {port}: motors enabled={self.enabled}; awaiting protocol/session feedback')

    def _write(self,kind,stamp,payload):
        frame=encode_frame(Frame(kind,self.session.session,self.sequence,stamp,payload))
        write_frame_once(self.serial.fileno(),frame)
        self.sequence=(self.sequence+1)&0xffffffff

    def fail(self,reason):
        if self.failed:return
        self.failed=True;self.session.fault=True
        try:self._write(STOP,0,b'')
        except (OSError,serial.SerialException,RuntimeError,ValueError):pass
        message=String();message.data=json.dumps(dict(fault=True,reason=str(reason),enabled=False))
        self.diag_pub.publish(message);self.get_logger().error(str(reason))

    def on_command(self,message):
        if self.failed:return
        try:
            self.matcher.command(contract.stamp_ns(message.header.stamp),message.name,message.position,message.effort,time.monotonic_ns())
            self.send_ready()
        except (KeyError,TypeError,ValueError,OSError,serial.SerialException,RuntimeError) as error:self.fail(error)

    def on_status(self,message):
        if self.failed:return
        try:
            self.matcher.status(json.loads(message.data),time.monotonic_ns());self.send_ready()
        except (KeyError,TypeError,ValueError,OSError,serial.SerialException,RuntimeError) as error:self.fail(error)

    def send_ready(self):
        if not self.session.acknowledged or self.session.feedback_time is None:return
        now=time.monotonic_ns();pair=self.matcher.take(now)
        if pair is None:return
        stamp,fields=pair
        self.session.command_sent(now)
        self._write(COMMAND,stamp,pack_command(**fields,enabled=self.enabled))

    def pump(self):
        if self.failed:return
        try:
            now=time.monotonic_ns()
            for frame in self.parser.feed(self.serial.read(min(4096,self.serial.in_waiting))):
                if frame.kind==HELLO_ACK:
                    self.session.ack(frame.session,struct.unpack('<I',frame.payload)[0],now)
                elif frame.kind==FEEDBACK:
                    fields=unpack_feedback(frame.payload)
                    if not self.session.feedback(frame.session,frame.sequence,fields['fault_flags'],frame.stamp_ns,now):continue
                    if self.base_stamp is None:self.base_stamp=frame.stamp_ns
                    stamp=frame.stamp_ns-self.base_stamp
                    if stamp<0:raise ValueError('MCU clock moved backwards')
                    message=JointState();message.header.stamp.sec,message.header.stamp.nanosec=divmod(stamp,1_000_000_000)
                    message.header.frame_id='joint_output';message.name=list(JOINT_NAMES)
                    message.position=fields['position'];message.velocity=fields['velocity'];message.effort=fields['effort']
                    self.feedback_pub.publish(message)
                    if frame.sequence%20==0:
                        status=String();status.data=json.dumps(dict(session=frame.session,sequence=frame.sequence,
                            mcu_stamp_ns=frame.stamp_ns,received_monotonic_ns=now,ack_sequence=fields['ack_sequence'],
                            armed=fields['armed'],fault_flags=fields['fault_flags'],enabled=self.enabled))
                        self.diag_pub.publish(status)
            self.session.check(now)
            if not self.session.acknowledged and now-self.started>2_000_000_000:
                raise TimeoutError('MCU HELLO_ACK timeout')
            self.send_ready()
        except (KeyError,TypeError,ValueError,OSError,serial.SerialException,RuntimeError) as error:self.fail(error)

    def destroy_node(self):
        if hasattr(self,'serial') and self.serial.is_open:
            try:self._write(STOP,0,b'')
            except (OSError,serial.SerialException,RuntimeError,ValueError):pass
            self.serial.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args);node=UsbGateway()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
