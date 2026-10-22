"""Reject a uniform crouch, unsupported holds and incomplete sit/rise traces."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def trace():
    t=np.arange(.005,14.001,.005)
    sit=(t>=5.5)&(t<8.)
    pos=np.zeros((len(t),3));pos[:,2]=.467;pos[sit,2]=.327
    quat=np.zeros((len(t),4));quat[:,0]=1.;quat[sit,0]=np.cos(.2);quat[sit,1]=np.sin(.2)
    q=np.tile([0.,-.55,1.1],(len(t),4));q[sit,8]=2.27;q[sit,11]=2.27
    support=np.zeros((len(t),4,3));support[:,:,2]=55.
    records=dict(state_stamp_ns=np.rint(t*1e9).astype(np.int64),position=pos,quaternion=quat,
        velocity=np.zeros((len(t),3)),angular_velocity=np.zeros((len(t),3)),
        joint_position=q,shank_support_force=support)
    history=[dict(mode=m) for m in ['STAND','LOWER','SETTLE_SIT','SIT','RISE','STABILIZE','STAND']]
    return records,history


def test_valid_four_foot_pitched_sit_and_rise_pass():
    from v15_dynamic_sit_metrics import dynamic_sit_verdict
    r,h=trace()
    assert dynamic_sit_verdict('dynamic_sit',r,h,True,.467,.327)['motion_success']


def test_level_crouch_is_rejected_even_with_correct_height_and_modes():
    from v15_dynamic_sit_metrics import dynamic_sit_verdict
    r,h=trace();r['quaternion'][:]=[1.,0.,0.,0.]
    verdict=dynamic_sit_verdict('dynamic_sit',r,h,True,.467,.327)
    assert not verdict['motion_success']
    assert 'seated_pitch_error_exceeds_0.08rad' in verdict['reasons']


def test_missing_support_or_unfinished_rise_is_rejected():
    from v15_dynamic_sit_metrics import dynamic_sit_verdict
    r,h=trace();r['shank_support_force'][:,0]=0.
    assert not dynamic_sit_verdict('dynamic_sit',r,h,True,.467,.327)['motion_success']
    r,h=trace()
    assert not dynamic_sit_verdict('dynamic_sit',r,h[:-2],False,.467,.327)['motion_success']
