"""ROS independent timestamp pairing, calibration gates and measured sensor freshness."""
import math

NAMES=tuple(f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HAA','HFE','KFE'))
STEP_NS=5_000_000


def _stamp(value):
    if type(value) is not int or value<0 or value%STEP_NS:
        raise ValueError('stamp must be on the nonnegative 5 ms grid')
    return value


def _vector(values,limit):
    result=list(values)
    if len(result)!=12 or any(not math.isfinite(v) or abs(v)>limit for v in result):
        raise ValueError('invalid finite 12-joint command')
    return result


class CommandMatcher:
    def __init__(self,max_pending=8,timeout_ns=50_000_000):
        if max_pending<1 or timeout_ns<=0:raise ValueError('positive buffer and timeout required')
        self.max_pending=max_pending;self.timeout_ns=timeout_ns
        self.commands={};self.statuses={};self.last=-1

    @property
    def pending_count(self):
        return len(set(self.commands)|set(self.statuses))

    def _trim(self):
        stamps=sorted(set(self.commands)|set(self.statuses))
        for stamp in stamps[:-self.max_pending]:
            self.commands.pop(stamp,None);self.statuses.pop(stamp,None)

    def command(self,stamp,names,position,effort,now_ns):
        stamp=_stamp(stamp)
        if stamp<=self.last:return
        if len(names)!=12 or set(names)!=set(NAMES) or len(set(names))!=12:
            raise ValueError('joint names must cover the twelve canonical joints')
        position=_vector(position,12.5);effort=_vector(effort,48.)
        p=dict(zip(names,position));e=dict(zip(names,effort))
        self.commands[stamp]=(dict(position=[p[n] for n in NAMES],effort=[e[n] for n in NAMES]),now_ns)
        self._trim()

    def status(self,data,now_ns):
        stamp=_stamp(data['stamp_ns'])
        if stamp<=self.last:return
        kp,kd=data['stiffness_nm_rad'],data['damping_nms_rad']
        if data.get('command_schema')!=2 or not math.isfinite(kp) or not math.isfinite(kd) or not 0<=kp<=300 or not 0<=kd<=8:
            raise ValueError('unsupported command schema or invalid gains')
        self.statuses[stamp]=(dict(kp=[float(kp)]*12,kd=[float(kd)]*12),now_ns)
        self._trim()

    def take(self,now_ns):
        common=set(self.commands)&set(self.statuses)
        if not common:return None
        stamp=max(common)
        command,ct=self.commands[stamp];status,st=self.statuses[stamp]
        if now_ns-min(ct,st)>self.timeout_ns or now_ns<max(ct,st):
            raise ValueError('command pair expired')
        self.last=stamp
        for buffer in (self.commands,self.statuses):
            for old in list(buffer):
                if old<=stamp:buffer.pop(old)
        return stamp,dict(**command,**status,velocity=[0.]*12)


def validate_hardware_config(config,enabled):
    if not enabled:return
    motors=config.get('motors',[])
    if config.get('calibration_verified') is not True or len(motors)!=12:
        raise ValueError('motor activation requires verified calibration for all twelve joints')
    if {m.get('joint') for m in motors}!=set(NAMES):
        raise ValueError('calibration must name all canonical joints')
    addresses=set()
    for motor in motors:
        ident=motor.get('id');bus=motor.get('bus',0)
        if type(ident) is not int or not 1<=ident<=255 or type(bus) is not int or bus<0:
            raise ValueError('calibration requires valid CAN bus/ID')
        if (bus,ident) in addresses:raise ValueError('duplicate motor CAN address')
        addresses.add((bus,ident))
        if motor.get('direction') not in (-1,1) or not isinstance(motor.get('zero_rad'),(int,float)) or not math.isfinite(motor['zero_rad']):
            raise ValueError('calibration requires direction and finite output-shaft zero_rad')


class SensorGate:
    def __init__(self):
        self.base=None;self.last=None;self.fault=False

    def step(self,feedback_stamp,now_ns,measurement_stamps):
        if self.fault:raise ValueError('sensor fault is latched; restart after diagnosis')
        _stamp(feedback_stamp)
        limits=dict(odom=75_000_000,depth=200_000_000,contacts=75_000_000)
        ready=all(n in measurement_stamps and -10_000_000<=now_ns-measurement_stamps[n]<=limit for n,limit in limits.items())
        if self.base is None:
            if not ready:return None
            self.base=feedback_stamp;self.last=feedback_stamp
            return 0
        if feedback_stamp<=self.last:return None
        if feedback_stamp!=self.last+STEP_NS:
            self.fault=True;raise ValueError('missing MCU feedback tick')
        logical=feedback_stamp-self.base
        if logical%(4*STEP_NS)==0 and not ready:
            self.fault=True;raise ValueError('stale or missing measured policy sensor')
        self.last=feedback_stamp
        return logical


class SessionGuard:
    def __init__(self,session,timeout_ns=50_000_000):
        if type(session) is not int or not 0<session<=0xffffffff or timeout_ns<=0:
            raise ValueError('valid nonzero session and timeout required')
        self.session=session;self.timeout_ns=timeout_ns;self.acknowledged=False
        self.ack_time=None;self.feedback_time=None;self.command_time=None;self.sequence=None;self.fault=False

    def ack(self,session,capabilities,now_ns):
        if session!=self.session or self.fault:return False
        if capabilities&1!=1:raise ValueError('MCU lacks full hybrid command capability')
        if self.acknowledged:return False
        self.acknowledged=True;self.ack_time=now_ns
        return True

    def feedback(self,session,sequence,fault_flags,stamp_ns,now_ns):
        if self.fault:raise ValueError('USB link fault is latched')
        if not self.acknowledged or session!=self.session:return False
        if self.sequence is not None and not 0<(sequence-self.sequence)&0xffffffff<0x80000000:return False
        self.check(now_ns)
        _stamp(stamp_ns)
        if fault_flags:
            self.fault=True;raise ValueError(f'MCU fault flags {fault_flags}')
        self.sequence=sequence;self.feedback_time=now_ns
        return True

    def command_sent(self,now_ns):
        self.check(now_ns)
        if not self.acknowledged or self.feedback_time is None:raise ValueError('fresh MCU feedback required before command')
        self.command_time=now_ns

    def check(self,now_ns):
        if self.fault:raise ValueError('USB link fault is latched')
        reference=self.feedback_time if self.feedback_time is not None else self.ack_time
        if reference is not None and now_ns-reference>self.timeout_ns:
            self.fault=True;raise ValueError('MCU feedback watchdog expired')
        if self.command_time is not None and now_ns-self.command_time>self.timeout_ns:
            self.fault=True;raise ValueError('ROS2 command watchdog expired')
