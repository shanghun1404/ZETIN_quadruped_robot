import importlib.util
from pathlib import Path
import numpy as np
import pytest

P = Path(__file__).parents[1] / 'v151_ros2_control' / 'motion.py'

def api():
    assert P.exists(), 'V15.1 transition controller is missing'
    spec = importlib.util.spec_from_file_location('v151_motion', P)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_rise_requires_measured_stability_before_walking():
    m = api().MotionSupervisor()
    m.request('walk', [0., .4, 0.])
    for k in range(251):
        out = m.step(k*.02, stable=False)
    assert out['mode'] == 'STABILIZE'
    assert out['walk_weight'] == 0
    for k in range(251, 350):
        out = m.step(k*.02, stable=True)
    assert out['mode'] == 'WALK'
    assert out['walk_weight'] == 1

def test_stop_ramps_command_then_requires_stability_before_sit():
    m = api().MotionSupervisor(initial_mode='STAND')
    m.request('walk', [0., .4, 0.])
    for k in range(151):
        m.step(k*.02, stable=True)
    m.request('sit')
    out = m.step(3.02, stable=False)
    assert out['mode'] == 'BRAKE'
    assert out['command'][1] > 0
    for k in range(152, 251):
        out = m.step(k*.02, stable=False)
    assert out['mode'] == 'BRAKE'
    assert out['height_fraction'] == 1
    for k in range(251, 550):
        out = m.step(k*.02, stable=True)
    assert out['mode'] == 'SIT'
    assert out['height_fraction'] == 0

def test_failure_timeout_and_invalid_time_are_visible():
    m = api().MotionSupervisor()
    m.request('stand')
    m.step(0, stable=False)
    out = m.step(20, stable=False)
    out = m.step(40, stable=False)
    assert out['mode'] == 'FAULT'
    with pytest.raises(ValueError):
        m.step(39, stable=True)

def test_torque_conversion_respects_speed_envelope():
    mod = api()
    q = np.zeros(12); dq = np.full(12, 49.)
    target, torque = mod.torque_position_target(np.full(12, 100.), q, dq)
    assert np.max(torque) <= .99 * 48 * .02 + 1e-10
    np.testing.assert_allclose(300*(target-q)-8*dq, torque, atol=1e-10)

def test_seated_label_requires_measured_low_pose():
    m=api().MotionSupervisor(initial_mode='STAND')
    m.request('sit')
    for k in range(300): out=m.step(k*.02,stable=True,seated_ready=False)
    assert out['mode']=='SETTLE_SIT'
    for k in range(300,330): out=m.step(k*.02,stable=False,seated_ready=True)
    assert out['mode']=='SIT'

def test_zero_velocity_request_preserves_last_walk_command_for_braking():
    m=api().MotionSupervisor(initial_mode='WALK')
    m.request('walk',[0.,.4,0.]);m.step(0,stable=False)
    m.request('stand',[0.,0.,0.])
    out=m.step(.02,stable=False)
    assert out['command'][1]==pytest.approx(.4)
    out=m.step(.52,stable=False)
    assert out['command'][1]==pytest.approx(.2)

def test_pending_timeout_starts_with_request_not_idle_stand():
    m=api().MotionSupervisor(initial_mode='STAND')
    m.step(0,stable=False);m.step(100,stable=False)
    m.request('walk',[0.,.4,0.])
    assert m.step(100.02,stable=False)['mode']=='STAND'
    assert m.step(113,stable=False)['mode']=='FAULT'

def test_hybrid_stand_uses_direct_effort_and_walk_preserves_prior_target():
    mod=api();q=np.linspace(-.1,.1,12);dq=np.linspace(-1,1,12);target=q+.05
    tau=np.tile([4.,8.,12.],4)
    stand=mod.hybrid_motor_command(0.,tau,target,q,dq)
    assert stand['kp']==0 and stand['kd']==0
    np.testing.assert_allclose(stand['effort'],tau)
    walk=mod.hybrid_motor_command(1.,tau,target,q,dq)
    old,_=mod.torque_position_target(300*(target-q)-8*dq,q,dq)
    np.testing.assert_allclose(walk['position'],old)
    np.testing.assert_allclose(walk['effort'],0)
    mix=mod.hybrid_motor_command(.4,tau,target,q,dq)
    actual=mix['kp']*(mix['position']-q)-mix['kd']*dq+mix['effort']
    np.testing.assert_allclose(actual,mix['bounded_torque'],atol=1e-12)

def test_stop_handover_can_start_while_moving_with_measured_support():
    m=api().MotionSupervisor(initial_mode='WALK')
    m.request('walk',[0.,.4,0.]);m.step(0,stable=False)
    m.request('stand',[0.,0.,0.]);m.step(.02,stable=False,support_ready=True)
    out=m.step(.34,stable=False,support_ready=True)
    assert out['mode']=='BLEND_STAND'
    assert out['command'][1]>0
    out=m.step(.84,stable=False,support_ready=True)
    assert 0<out['walk_weight']<1
    assert 0<out['command'][1]<.4
    out=m.step(1.4,stable=False,support_ready=True)
    assert out['mode']=='STABILIZE'
    assert out['walk_weight']==0

def test_feedforward_leaves_drive_budget_with_hard_total_bound():
    mod=api();ff=np.linspace(-22,22,12)
    cap=mod.drive_force_budget(ff)
    assert np.all(cap+np.abs(ff)<=mod.LIMITS+1e-12)
    np.testing.assert_allclose(mod.drive_force_budget(np.zeros(12)),mod.LIMITS)
    with pytest.raises(ValueError): mod.drive_force_budget(np.full(12,100.))
