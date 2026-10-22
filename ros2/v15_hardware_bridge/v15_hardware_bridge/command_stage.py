"""Measured-joint command generation independent of the 50 Hz planner process."""
import copy
import math
import numpy as np
import torch
from v14_ros2_control import contract
from v14_ros2_control.joint_command import JointCommandFilter
from v151_ros2_control.motion import CANONICAL, hybrid_motor_command
from .core import _stamp


def _finite_tree(value):
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError('plan keys must be strings')
        for item in value.values():_finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:_finite_tree(item)
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):raise ValueError('non-finite plan value')
    elif value is not None and not isinstance(value, (str, bool)):
        raise ValueError('unsupported plan value')


def _vector(value, size):
    a=np.asarray(value,dtype=float)
    if a.shape!=(size,) or not np.isfinite(a).all():
        raise ValueError(f'expected {size} finite values')
    return a


class PlanCommandStage:
    def __init__(self, max_plan_age_ns=50_000_000, max_pending=8):
        if type(max_plan_age_ns) is not int or max_plan_age_ns<=0 or type(max_pending) is not int or max_pending<1:
            raise ValueError('positive plan lifetime and buffer required')
        self.max_age=max_plan_age_ns;self.max_pending=max_pending
        self.plans={};self.filter=JointCommandFilter();self.generation=None
        self.last_stamp=-1;self.latest_joint_stamp=-1;self.last_result=None;self.last_plan=None
        self.to_canonical=[contract.JOINT_NAMES.index(n) for n in CANONICAL]
        self.to_policy=[CANONICAL.index(n) for n in contract.JOINT_NAMES]

    @property
    def pending_count(self):return len(self.plans)

    def update(self, plan, received_ns):
        _finite_tree(plan)
        stamp=_stamp(plan['source_stamp_ns'])
        if stamp%20_000_000 or plan.get('plan_schema')!=1:
            raise ValueError('unsupported plan schema or 50 Hz source stamp')
        generation=plan['generation']
        if type(generation) is not int or generation<0:
            raise ValueError('invalid plan generation')
        _vector(plan['raw_target'],12);_vector(plan['standing_torque'],12)
        status=plan['status'];weight=status['walk_weight'];height=status['height_fraction']
        if status['mode'] not in ('SIT','SETTLE_SIT','RISE','LOWER','STABILIZE','STAND','BLEND_WALK','WALK','BRAKE','BLEND_STAND'):
            raise ValueError('invalid or faulted planner mode')
        if not 0.<=weight<=1. or not 0.<=height<=1.:
            raise ValueError('invalid plan blend weight or height fraction')
        _vector(status['command'],3)
        if type(received_ns) is not int or received_ns<0:
            raise ValueError('invalid plan receipt clock')
        if stamp in self.plans or (self.last_plan is not None and stamp<=self.last_plan[0]['source_stamp_ns']):
            return False
        self.plans[stamp]=(copy.deepcopy(plan),received_ns)
        for old in sorted(self.plans)[:-self.max_pending]:self.plans.pop(old)
        return True

    def _current(self, stamp):
        eligible=[s for s in self.plans if s<=stamp]
        return self.plans[max(eligible)] if eligible else None

    def _fresh(self, plan, received_ns, stamp, now_ns):
        if now_ns<received_ns or now_ns-received_ns>self.max_age or stamp-plan['source_stamp_ns']>self.max_age:
            raise TimeoutError('planner result expired or stale')

    def check_deadline(self, now_ns):
        if self.last_result is None:return
        self._fresh(*self.last_plan,self.latest_joint_stamp,now_ns)

    def command(self, stamp, names, position, velocity, now_ns):
        stamp=_stamp(stamp)
        if stamp<self.latest_joint_stamp:return None
        self.latest_joint_stamp=stamp
        if stamp==self.last_stamp:
            self._fresh(*self.last_plan,stamp,now_ns)
            return self.last_result
        current=self._current(stamp)
        if current is None:
            if self.last_result is not None:raise TimeoutError('no nonfuture plan; previous result expired')
            return None
        plan,received_ns=current
        try:self._fresh(plan,received_ns,stamp,now_ns)
        except TimeoutError:
            if self.last_result is None:return None
            raise
        q=_vector(contract.ordered(names,position),12)
        dq=_vector(contract.ordered(names,velocity),12)
        qt=torch.tensor(q,dtype=torch.float32);dqt=torch.tensor(dq,dtype=torch.float32)
        if self.generation!=plan['generation']:
            self.filter.reset(qt);self.generation=plan['generation']
        self.filter.set_target(torch.tensor(plan['raw_target'],dtype=torch.float32))
        target=self.filter.step(qt,dqt).numpy()
        status=copy.deepcopy(plan['status'])
        motor=hybrid_motor_command(status['walk_weight'],plan['standing_torque'],target[self.to_canonical],
                                   q[self.to_canonical],dq[self.to_canonical])
        status.update(stamp_ns=stamp,plan_stamp_ns=plan['source_stamp_ns'],
                      plan_age_ms=(stamp-plan['source_stamp_ns'])/1e6,
                      command_schema=2,stiffness_nm_rad=motor['kp'],damping_nms_rad=motor['kd'],
                      requested_torque=motor['requested_torque'].tolist(),bounded_torque=motor['bounded_torque'].tolist())
        self.last_result=dict(stamp_ns=stamp,names=list(contract.JOINT_NAMES),
                              position=motor['position'][self.to_policy].tolist(),
                              effort=motor['effort'][self.to_policy].tolist(),status=status)
        self.last_stamp=stamp;self.last_plan=current
        return self.last_result
