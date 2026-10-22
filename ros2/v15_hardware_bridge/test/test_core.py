"""Failures caught: mixed timestamps, delayed replay, unsafe arming, missing sensors."""
import importlib.util
import math
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1]/'v15_hardware_bridge/core.py'


def core():
    assert PATH.is_file(), 'hardware core is not implemented'
    spec = importlib.util.spec_from_file_location('hardware_core', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NAMES = tuple(f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HAA','HFE','KFE'))


def test_atomic_pair_waits_for_same_stamp_and_orders_by_name():
    c = core(); matcher = c.CommandMatcher()
    matcher.command(0, list(reversed(NAMES)), list(range(12)), [0.]*12, 100)
    matcher.status(dict(stamp_ns=5_000_000,command_schema=2,stiffness_nm_rad=300.,damping_nms_rad=8.),100)
    assert matcher.take(110) is None
    matcher.status(dict(stamp_ns=0,command_schema=2,stiffness_nm_rad=150.,damping_nms_rad=4.),100)
    stamp, data = matcher.take(110)
    assert stamp == 0
    assert data['position'] == list(reversed(range(12)))
    assert data['kp'] == [150.]*12
    assert data['kd'] == [4.]*12
    assert data['velocity'] == [0.]*12
    assert matcher.take(110) is None


def test_old_complete_pair_cannot_renew_command_after_new_pair():
    c = core(); matcher = c.CommandMatcher()
    for stamp in (0,5_000_000):
        matcher.command(stamp,NAMES,[0.]*12,[0.]*12,100)
        matcher.status(dict(stamp_ns=stamp,command_schema=2,stiffness_nm_rad=0.,damping_nms_rad=0.),100)
    assert matcher.take(110)[0] == 5_000_000
    assert matcher.take(110) is None
    matcher.command(0,NAMES,[0.]*12,[0.]*12,120)
    matcher.status(dict(stamp_ns=0,command_schema=2,stiffness_nm_rad=0.,damping_nms_rad=0.),120)
    assert matcher.take(120) is None


def test_delayed_or_invalid_command_is_rejected():
    c = core(); matcher = c.CommandMatcher()
    matcher.command(0,NAMES,[0.]*12,[0.]*12,0)
    matcher.status(dict(stamp_ns=0,command_schema=2,stiffness_nm_rad=0.,damping_nms_rad=0.),0)
    with pytest.raises(ValueError,match='expired'):
        matcher.take(50_000_001)
    with pytest.raises(ValueError):
        matcher.command(1,NAMES,[0.]*12,[0.]*12,100)
    with pytest.raises(ValueError):
        matcher.command(0,NAMES,[math.nan]*12,[0.]*12,100)
    with pytest.raises(ValueError):
        matcher.status(dict(stamp_ns=0,command_schema=1,stiffness_nm_rad=300.,damping_nms_rad=8.),100)


def test_pending_messages_are_bounded():
    matcher = core().CommandMatcher(max_pending=4)
    for step in range(100):
        matcher.command(step*5_000_000,NAMES,[0.]*12,[0.]*12,step)
    assert matcher.pending_count <= 4


def test_arm_requires_complete_calibration():
    c=core()
    c.validate_hardware_config({},False)
    with pytest.raises(ValueError,match='calibration'):
        c.validate_hardware_config({},True)
    config=dict(calibration_verified=True,motors=[dict(joint=n,id=i+1,direction=1,zero_rad=0.) for i,n in enumerate(NAMES)])
    c.validate_hardware_config(config,True)
    config['motors'][1]['id']=1
    with pytest.raises(ValueError):c.validate_hardware_config(config,True)


def test_sensor_freshness_and_first_ready_tick_then_gap_fault():
    sync = core().SensorGate()
    assert sync.step(10_000_000,1_000_000_000,{}) is None
    times=dict(odom=990_000_000,depth=900_000_000,contacts=980_000_000)
    assert sync.step(15_000_000,1_000_000_000,times) == 0
    assert sync.step(20_000_000,1_000_000_000,times) == 5_000_000
    assert sync.step(20_000_000,1_000_000_000,times) is None
    with pytest.raises(ValueError,match='missing'):
        sync.step(30_000_000,1_000_000_000,times)
    with pytest.raises(ValueError,match='latched'):
        sync.step(35_000_000,1_000_000_000,times)


@pytest.mark.parametrize('times',[
    dict(odom=900_000_000,depth=900_000_000,contacts=980_000_000),
    dict(odom=1_020_000_000,depth=900_000_000,contacts=980_000_000),
    dict(odom=990_000_000,depth=700_000_000,contacts=980_000_000),
])
def test_stale_or_future_sensor_cannot_start_controller(times):
    gate=core().SensorGate()
    assert gate.step(0,1_000_000_000,times) is None


def test_link_session_ignores_foreign_and_replayed_feedback_without_renewing_watchdog():
    c=core();assert hasattr(c,'SessionGuard'),'USB session guard is missing'
    link=c.SessionGuard(123)
    assert not link.ack(999,1,0)
    assert link.ack(123,1,0)
    assert not link.feedback(999,0,0,0,1)
    assert link.feedback(123,0,0,0,10)
    assert not link.feedback(123,0,0,0,40_000_000)
    with pytest.raises(ValueError,match='feedback'):
        link.check(50_000_011)


def test_lost_commands_or_board_fault_latches_link():
    c=core();assert hasattr(c,'SessionGuard'),'USB session guard is missing'
    link=c.SessionGuard(1);link.ack(1,1,0);link.feedback(1,0,0,0,0);link.command_sent(0)
    link.feedback(1,1,0,5_000_000,40_000_000)
    with pytest.raises(ValueError,match='command'):link.check(50_000_001)
    with pytest.raises(ValueError,match='latched'):link.check(50_000_002)
    board=c.SessionGuard(2);board.ack(2,1,0)
    with pytest.raises(ValueError,match='MCU fault'):board.feedback(2,0,4,0,0)


def test_sequence_wrap_is_accepted_but_backward_feedback_is_ignored():
    c=core();assert hasattr(c,'SessionGuard'),'USB session guard is missing'
    link=c.SessionGuard(1);link.ack(1,1,0)
    assert link.feedback(1,0xffffffff,0,0,0)
    assert link.feedback(1,0,0,5_000_000,1)
    assert not link.feedback(1,0xfffffffe,0,10_000_000,2)


def test_late_fresh_feedback_cannot_hide_watchdog_gap():
    link=core().SessionGuard(1);link.ack(1,1,0);link.feedback(1,0,0,0,0)
    with pytest.raises(ValueError,match='feedback'):
        link.feedback(1,1,0,5_000_000,60_000_000)
