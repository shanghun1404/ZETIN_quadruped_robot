import numpy as np
import pytest

from stair_task import analyze, ground_rise_from_center


def recording(xy, completed=True):
    xy=np.asarray(xy,float);n=len(xy)
    return dict(position=np.c_[xy,np.full(n,.464)],
                quaternion=np.tile([1.,0.,0.,0.],(n,1)),
                gaps=np.full((n,24),.01),same_leg_thigh_foot_gap_m=np.full((n,4),.2),
                contact_pair_force=np.zeros((n,12))), dict(scenario='stairs_up',completed=completed,
                    failure=None if completed else 'fallen',task_success=completed,
                    motion_success=completed,collision_checks_pass=True,
                    geometry=dict(step_height_m=1.1/7,full_rise_m=1.1))


def test_ground_matches_bottom_six_rings_and_outer_border():
    radii=[0.,1.19,1.35,1.65,1.95,2.25,2.55,2.85,3.15,3.6]
    xy=np.c_[np.zeros(len(radii)),radii]
    expected=np.array([0,0,1,2,3,4,5,6,7,7])*1.1/7
    np.testing.assert_allclose(ground_rise_from_center(xy,1.1/7),expected,atol=1e-12)
    # Square rings must have the same height on their side and diagonal.
    np.testing.assert_allclose(ground_rise_from_center(xy[:,::-1],1.1/7),expected,atol=1e-12)
    np.testing.assert_allclose(ground_rise_from_center(np.c_[radii,radii],1.1/7),expected,atol=1e-12)


def test_stationary_raw_success_is_overridden_and_body_height_is_not_ground():
    arrays,raw=recording([[10.,20.],[10.,20.]])
    arrays['position'][-1,2]=2.
    result=analyze(arrays,raw)
    assert result['raw_task_success'] is True
    assert result['posthoc_directed_stair_task_success'] is False
    assert result['posthoc_full_flight_task_success'] is False
    assert result['path']['per_environment'][0]['final_ground_change_m']==0.


def test_forward_full_flight_uses_initial_tile_center_and_keeps_raw_verdict():
    arrays,raw=recording([[10.,20.],[10.,21.35],[10.,22.25],[10.,23.6]])
    result=analyze(arrays,raw)
    assert result['posthoc_directed_stair_task_success'] is True
    assert result['posthoc_full_flight_task_success'] is True
    assert result['terrain_center_xy_inferred_from_first_root_sample_m']==[10.,20.]
    assert raw['task_success'] is True


@pytest.mark.parametrize('case',['cross_track','heading','failed','collision','returned'])
def test_full_height_alone_cannot_bypass_other_existing_gates(case):
    arrays,raw=recording([[0.,0.],[0.,2.25],[0.,3.7]])
    if case=='cross_track': arrays['position'][1,0]=.61
    elif case=='heading': arrays['quaternion'][1]=[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
    elif case=='failed': raw['completed']=False;raw['failure']='fallen'
    elif case=='collision': arrays['same_leg_thigh_foot_gap_m'][1,0]=-.001
    elif case=='returned': arrays['position'][-1,:2]=[0.,0.]
    result=analyze(arrays,raw)
    assert result['posthoc_full_flight_task_success'] is False


def test_changed_or_unknown_geometry_is_rejected():
    arrays,raw=recording([[0.,0.],[0.,3.6]])
    raw['geometry']['full_rise_m']=.9
    with pytest.raises(ValueError,match='geometry'): analyze(arrays,raw)
