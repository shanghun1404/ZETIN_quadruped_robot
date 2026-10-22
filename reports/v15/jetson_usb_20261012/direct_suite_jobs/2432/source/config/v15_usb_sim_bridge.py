"""Same CAD physics, with joint feedback and hybrid commands through Python/C/PTY."""
import json
import os
from pathlib import Path
import numpy as np
from v14_ros2_control import contract
from v15_cad_ros2_bridge import V15CadBridge
from v15_usb_roundtrip import CodecRoundtrip
from v15_hardware_bridge.protocol import Frame,COMMAND,FEEDBACK,JOINT_NAMES,pack_command,unpack_command,pack_feedback,unpack_feedback


class UsbCadBridge(V15CadBridge):
    def __init__(self,*args,**kwargs):
        self.usb=CodecRoundtrip(os.environ['V15_USB_CODEC'])
        self.usb_sequence=0;self.usb_session=123
        self.case=kwargs.get('namespace','').rsplit('/',1)[-1]
        self.to_canonical=[contract.JOINT_NAMES.index(name) for name in JOINT_NAMES]
        self.to_policy=[JOINT_NAMES.index(name) for name in contract.JOINT_NAMES]
        super().__init__(*args,**kwargs)

    def exchange(self,kind,stamp,payload):
        frame=Frame(kind,self.usb_session,self.usb_sequence,stamp,payload)
        self.usb_sequence+=1
        return self.usb.exchange(frame).payload

    def _publish_joint_state(self,sec,nanosec):
        data=self._asset.data;indices=self.to_canonical
        values=[(field[0,self._joint_ids]*self.sign).cpu().numpy()[indices].tolist()
                for field in (data.joint_pos,data.joint_vel,data.applied_torque)]
        fields=unpack_feedback(self.exchange(FEEDBACK,sec*1_000_000_000+nanosec,pack_feedback(*values)))
        message=self._header(self._msg['JointState'](),sec,nanosec);message.name=list(contract.JOINT_NAMES)
        for name in ('position','velocity','effort'):
            setattr(message,name,np.asarray(fields[name])[self.to_policy].tolist())
        self._publish_input(self._state_pub,message)
        clock=self._msg['Clock']();clock.clock.sec,clock.clock.nanosec=sec,nanosec;self._clock_pub.publish(clock)

    def _wait_for_reply(self,stamp):
        super()._wait_for_reply(stamp)
        q,tau=self._commands[stamp];sign=self.sign.cpu().numpy();status=self._statuses[stamp]
        fields=unpack_command(self.exchange(COMMAND,stamp,pack_command(
            (np.asarray(q)*sign)[self.to_canonical],np.zeros(12),
            [status['stiffness_nm_rad']]*12,[status['damping_nms_rad']]*12,
            (np.asarray(tau)*sign)[self.to_canonical],enabled=False)))
        self._commands[stamp]=((np.asarray(fields['position'])[self.to_policy]*sign).tolist(),
                               (np.asarray(fields['effort'])[self.to_policy]*sign).tolist())
        status['stiffness_nm_rad']=float(fields['kp'][0]);status['damping_nms_rad']=float(fields['kd'][0])

    def close_usb(self):
        self.usb.close()
        folder=Path(os.environ['V15_USB_RESULTS'])/self.case
        (folder/'usb_transport.json').write_text(json.dumps(self.usb.stats,indent=2)+'\n')
