import numpy as np
from v15_cad_suite_metrics import estimated_torque, torque_summary, standing_verdict


def test_pd_clip_is_separate_from_feedforward_and_respects_sign_mapping():
    q=np.zeros(12);dq=np.zeros(12);target=np.ones(12)
    feedforward=np.tile([10.,-12.,20.],4)
    peak=np.tile([24.8,48.,48.],4)
    budget=peak-abs(feedforward)
    tau=estimated_torque(q,dq,target,300.,8.,budget,feedforward)
    np.testing.assert_allclose(tau,budget+feedforward)
    sign=np.tile([-1.,1.,-1.],4)
    native=estimated_torque(q*sign,dq*sign,target*sign,300.,8.,budget,feedforward*sign)
    np.testing.assert_allclose(native*sign,tau)
    assert np.max(abs(tau)-peak)<1e-10


def test_rated_rms_is_distinct_from_instantaneous_and_peak_limits():
    tau=np.zeros((200,12));tau[0,0]=24.;tau[:,1]=19.
    out=torque_summary(tau,np.zeros_like(tau))
    assert out['rated_exceed_fraction'][0]==.005
    assert not out['rms_exceeds_rated'][0]
    assert out['rms_exceeds_rated'][1]
    assert not out['simulator_peak_exceeded']
    assert not out['motor_datasheet_peak_exceeded']
    assert out['ideal_current_rms_a'][1]==19./1.44


def test_standing_excursion_cannot_pass_by_returning_to_start():
    t=np.linspace(.005,12.,2400)
    position=np.zeros((len(t),3));position[:,2]=.467
    position[:,0]=.075*(1-np.cos(2*np.pi*t/12.))
    velocity=np.zeros_like(position);velocity[:,0]=.075*2*np.pi/12.*np.sin(2*np.pi*t/12.)
    gravity=np.tile([0.,0.,-1.],(len(t),1))
    records=dict(state_stamp_ns=(t*1e9).astype(np.int64),position=position,velocity=velocity,gravity=gravity)
    verdict=standing_verdict('standing_hold',records,[dict(mode='STAND')],True,.467,.367)
    assert not verdict['motion_success']
    assert 'standing_drift_exceeds_0.10m' in verdict['reasons']


def test_early_standing_failure_retains_explicit_reason():
    records=dict(state_stamp_ns=np.array([5_000_000]),position=np.array([[0.,0.,.46]]),
                 velocity=np.zeros((1,3)),gravity=np.array([[0.,0.,-1.]]))
    verdict=standing_verdict('standing_hold',records,[],False,.46,.36)
    assert not verdict['motion_success']
    assert 'no_settled_standing_samples' in verdict['reasons']
