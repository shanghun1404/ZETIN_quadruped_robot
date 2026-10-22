"""Separate planning must preserve motor math and reject invalid old plans."""
import copy
import importlib.util
import numpy as np
import pytest
import torch
from v14_ros2_control import contract
from v14_ros2_control.joint_command import JointCommandFilter
from v151_ros2_control.motion import CANONICAL, hybrid_motor_command


def stage():
    assert importlib.util.find_spec('v15_hardware_bridge.command_stage') is not None, 'Separate command stage is missing'
    from v15_hardware_bridge.command_stage import PlanCommandStage
    return PlanCommandStage()


def plan(stamp=0, generation=0, weight=.5):
    q = np.array([contract.joint_default(n) for n in contract.JOINT_NAMES])
    return dict(plan_schema=1, source_stamp_ns=stamp, generation=generation,
                raw_target=(q+.015).tolist(), standing_torque=np.linspace(-1., 1., 12).tolist(),
                status=dict(mode='BLEND_WALK', walk_weight=weight, height_fraction=1.,
                            command=[0.,.4,0.], planner_compute_ms=8.))


def measured(s, stamp, now):
    q = np.array([contract.joint_default(n) for n in contract.JOINT_NAMES])
    return s.command(stamp, contract.JOINT_NAMES, q, np.zeros(12), now)


def test_no_command_until_a_nonfuture_plan_is_available():
    s=stage()
    assert measured(s,0,0) is None
    s.update(plan(20_000_000),0)
    assert measured(s,5_000_000,5_000_000) is None
    s.update(plan(),5_000_000)
    assert measured(s,10_000_000,10_000_000)['status']['plan_stamp_ns']==0


@pytest.mark.parametrize('weight',[0.,.25,1.])
def test_command_matches_existing_filter_and_hybrid_motor_law(weight):
    s=stage();p=plan(weight=weight);s.update(p,0)
    reference=JointCommandFilter()
    q=torch.tensor([contract.joint_default(n) for n in contract.JOINT_NAMES]);dq=torch.zeros(12)
    reference.reset(q);reference.set_target(torch.tensor(p['raw_target'],dtype=torch.float32))
    canonical=[contract.JOINT_NAMES.index(n) for n in CANONICAL]
    policy=[CANONICAL.index(n) for n in contract.JOINT_NAMES]
    for stamp in range(0,20_000_000,5_000_000):
        target=reference.step(q,dq).numpy()
        expected=hybrid_motor_command(weight,p['standing_torque'],target[canonical],q.numpy()[canonical],dq.numpy()[canonical])
        result=s.command(stamp,contract.JOINT_NAMES,q.numpy(),dq.numpy(),stamp)
        np.testing.assert_allclose(result['position'],expected['position'][policy],atol=1e-7)
        np.testing.assert_allclose(result['effort'],expected['effort'][policy],atol=1e-7)
        assert result['status']['stamp_ns']==stamp
        assert result['status']['stiffness_nm_rad']==expected['kp']
        assert result['status']['damping_nms_rad']==expected['kd']


def test_duplicate_feedback_replays_without_advancing_filter():
    s=stage();s.update(plan(),0)
    first=measured(s,0,0)
    assert measured(s,0,1_000_000)==first


@pytest.mark.parametrize('watchdog',[False,True])
def test_unused_new_plan_cannot_refresh_cached_commands(watchdog):
    s=stage();s.update(plan(),0);measured(s,20_000_000,0)
    s.update(plan(20_000_000),45_000_000)
    with pytest.raises(TimeoutError,match='expired|stale'):
        if watchdog:s.check_deadline(60_000_000)
        else:measured(s,20_000_000,60_000_000)


def test_startup_waits_for_fresh_plan_without_publishing_old_history():
    s=stage();s.update(plan(),100_000_000)
    assert measured(s,200_000_000,100_000_000) is None
    s.update(plan(200_000_000),101_000_000)
    assert measured(s,205_000_000,102_000_000)['status']['plan_stamp_ns']==200_000_000


def test_expired_plan_and_replayed_plan_cannot_keep_commands_alive():
    s=stage();p=plan();s.update(p,0);measured(s,0,0)
    assert s.update(p,40_000_000) is False
    with pytest.raises((ValueError,TimeoutError),match='expired|stale'):
        measured(s,20_000_000,51_000_000)


def test_evicted_used_plan_cannot_be_reinserted_with_a_new_receipt():
    s=stage();s.max_pending=1;p=plan();s.update(p,0);measured(s,0,0)
    s.update(plan(20_000_000),40_000_000)
    assert s.update(p,49_000_000) is False
    with pytest.raises(TimeoutError,match='expired|stale'):
        s.check_deadline(51_000_000)


def test_source_age_expires_even_if_receipt_was_recent():
    s=stage();s.update(plan(),100_000_000);measured(s,0,100_000_000)
    with pytest.raises((ValueError,TimeoutError),match='expired|stale'):
        measured(s,60_000_000,101_000_000)


def test_future_plan_does_not_refresh_running_watchdog():
    s=stage();s.update(plan(),0);measured(s,0,0)
    s.update(plan(100_000_000),49_000_000)
    with pytest.raises((ValueError,TimeoutError),match='expired|stale'):
        s.check_deadline(51_000_000)


@pytest.mark.parametrize('field,value', [('raw_target',[float('nan')]*12),('standing_torque',[float('inf')]*12),
                                        ('generation',-1),('source_stamp_ns',5_000_000),('plan_schema',99)])
def test_malformed_plans_fail_closed(field,value):
    s=stage();p=plan();p[field]=value
    with pytest.raises(ValueError):s.update(p,0)


def test_fault_mode_fails_closed():
    s=stage();p=plan();p['status']['mode']='FAULT'
    with pytest.raises(ValueError):s.update(p,0)


def test_plan_snapshot_is_independent_of_publishers_mutable_input():
    s=stage();p=plan();s.update(p,0);p['raw_target'][:]=[float('nan')]*12
    assert np.isfinite(measured(s,0,0)['position']).all()


def test_pending_plans_are_bounded():
    s=stage()
    for i in range(100):s.update(plan(i*20_000_000),i)
    assert s.pending_count<=8
