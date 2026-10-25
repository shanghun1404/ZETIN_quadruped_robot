"""Simulator-independent transition rules and motor command conversion (SI units)."""
import numpy as np

CANONICAL = tuple(f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HAA','HFE','KFE'))
LIMITS = np.tile([24.8, 48., 48.], 4)

def smooth(value):
    x = np.clip(value, 0., 1.)
    return x*x*x*(10+x*(-15+6*x))

def posture_forward_offset(height_fraction, stand_shift, crouch_shift, elapsed_s):
    """Height-dependent body shift with a one-second smooth startup (meters)."""
    if (not np.isfinite([height_fraction,stand_shift,crouch_shift,elapsed_s]).all()
            or not 0. <= height_fraction <= 1. or elapsed_s < 0.):
        raise ValueError('Expected finite shifts, height fraction in [0,1], nonnegative time')
    target=crouch_shift+(stand_shift-crouch_shift)*height_fraction
    return float(smooth(elapsed_s)*target)

def torque_position_target(torque, position, velocity, fraction=.99):
    torque, position, velocity = (np.asarray(v, dtype=float) for v in (torque, position, velocity))
    if any(v.shape != (12,) or not np.isfinite(v).all() for v in (torque,position,velocity)):
        raise ValueError('expected 12 finite joint values')
    upper = fraction*LIMITS*np.clip(1-velocity/50., 0, 1)
    lower = fraction*LIMITS*np.clip(-1-velocity/50., -1, 0)
    bounded = np.clip(torque, lower, upper)
    return position+(bounded+8*velocity)/300, bounded

class MotionSupervisor:
    """Measured readiness gates all handovers; elapsed time alone cannot authorize them."""
    def __init__(self, initial_mode='SIT', rise_s=3., blend_s=1., settle_s=.4, timeout_s=12.):
        if initial_mode not in ('SIT','STAND','WALK'):
            raise ValueError('invalid initial mode')
        self.mode=initial_mode; self.requested=initial_mode.lower()
        self.command=np.zeros(3); self.last_command=np.zeros(3); self.issued_command=np.zeros(3); self.pending_since=None
        self.rise_s=rise_s; self.blend_s=blend_s; self.settle_s=settle_s; self.timeout_s=timeout_s
        self.entered=0.; self.previous=-1.; self.stable_since=None; self.seated_since=None; self.events=[]

    def request(self, mode, command=None):
        if mode not in ('sit','stand','walk'):
            raise ValueError('mode must be sit, stand, or walk')
        if mode!=self.requested: self.pending_since=None
        self.requested=mode
        if command is not None:
            value=np.asarray(command,dtype=float)
            if value.shape!=(3,) or not np.isfinite(value).all():
                raise ValueError('command must be finite [vx,vy,wz]')
            self.command=value

    def _enter(self, mode, t):
        self.events.append(dict(time_s=t, previous=self.mode, mode=mode))
        self.mode=mode; self.entered=t; self.stable_since=None

    def step(self, t, *, stable, seated_ready=None, support_ready=None):
        if not np.isfinite(t) or t < self.previous:
            raise ValueError('time must increase monotonically')
        self.previous=t
        if stable:
            if self.stable_since is None: self.stable_since=t
        else: self.stable_since=None
        ready=self.stable_since is not None and t-self.stable_since >= self.settle_s-1e-8
        if seated_ready is None: seated_ready=stable
        if seated_ready:
            if self.seated_since is None: self.seated_since=t
        else: self.seated_since=None
        seat_ready=self.seated_since is not None and t-self.seated_since>=self.settle_s-1e-8
        if self.mode=='STAND' and self.requested!='stand':
            if self.pending_since is None: self.pending_since=t
        else: self.pending_since=None
        elapsed=t-self.entered
        if self.mode=='SIT' and self.requested!='sit': self._enter('RISE',t)
        elif self.mode=='RISE' and elapsed>=self.rise_s: self._enter('STABILIZE',t)
        elif self.mode=='STABILIZE' and ready: self._enter('STAND',t)
        elif self.mode=='STAND' and ready:
            if self.requested=='walk': self._enter('BLEND_WALK',t)
            elif self.requested=='sit': self._enter('LOWER',t)
        elif self.mode=='BLEND_WALK' and elapsed>=self.blend_s: self._enter('WALK',t)
        elif self.mode=='WALK' and self.requested!='walk':
            self.last_command=self.issued_command.copy(); self._enter('BRAKE',t)
        elif self.mode=='BRAKE' and elapsed>=.3 and (ready if support_ready is None else support_ready):
            self.last_command=self.issued_command.copy(); self._enter('BLEND_STAND',t)
        elif self.mode=='BLEND_STAND' and elapsed>=self.blend_s: self._enter('STABILIZE',t)
        elif self.mode=='LOWER' and elapsed>=self.rise_s: self._enter('SETTLE_SIT',t)
        elif self.mode=='SETTLE_SIT' and seat_ready: self._enter('SIT',t)
        elif ((self.mode in ('STABILIZE','BRAKE','SETTLE_SIT') and elapsed>self.timeout_s)
              or (self.pending_since is not None and t-self.pending_since>self.timeout_s)): self._enter('FAULT',t)
        elapsed=t-self.entered
        height=1.; weight=0.; command=np.zeros(3)
        if self.mode in ('SIT','SETTLE_SIT'): height=0.
        elif self.mode=='RISE': height=float(smooth(elapsed/self.rise_s))
        elif self.mode=='LOWER': height=1-float(smooth(elapsed/self.rise_s))
        elif self.mode=='BLEND_WALK':
            weight=float(smooth(elapsed/self.blend_s)); command=self.command*weight
        elif self.mode=='WALK': weight=1.; command=self.command.copy()
        elif self.mode=='BRAKE': weight=1.; command=self.last_command*(1-float(smooth(elapsed)))
        elif self.mode=='BLEND_STAND':
            weight=1-float(smooth(elapsed/self.blend_s)); command=self.last_command*weight
        self.issued_command=command.copy()
        return dict(mode=self.mode, walk_weight=weight, height_fraction=height, command=command.tolist())

def hybrid_motor_command(weight, standing_torque, walking_target, position, velocity):
    """MIT-style PD plus feedforward; pure walking exactly preserves the V15 target."""
    if not np.isfinite(weight) or not 0<=weight<=1: raise ValueError('weight must be in [0,1]')
    q,dq,target,stand=(np.asarray(x,dtype=float) for x in (position,velocity,walking_target,standing_torque))
    walking=300*(target-q)-8*dq
    requested=(1-weight)*stand+weight*walking
    guarded_target,bounded=torque_position_target(requested,q,dq)
    if weight==1.:
        target=guarded_target;feedforward=np.zeros(12)
    elif weight==0.:
        target=q.copy();feedforward=bounded
    else:
        feedforward=np.clip((1-weight)*stand+bounded-requested,-LIMITS,LIMITS)
        target=q+(bounded-feedforward+8*weight*dq)/(300*weight)
    return dict(position=target,effort=feedforward,kp=300*weight,kd=8*weight,
                requested_torque=requested,bounded_torque=bounded)


def drive_force_budget(feedforward):
    """Allocate drive maxForce so drive plus separate effort cannot exceed LIMITS."""
    ff=np.asarray(feedforward,dtype=float)
    if ff.shape!=(12,) or not np.isfinite(ff).all() or np.any(np.abs(ff)>LIMITS+1e-8):
        raise ValueError('feedforward must be finite and within joint effort limits')
    return np.maximum(0.,LIMITS-np.abs(ff))
