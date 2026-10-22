import numpy as np
import pytest


def test_metrics_keep_world_y_progress_body_velocity_error_and_motor_units():
    from compare_ros2 import metrics_from_arrays

    a = dict(position=np.array([[0,0,.4],[0,.002,.4],[0,.004,.4],[0,.006,.4]]),
             velocity=np.tile([0,.3,0.],(4,1)), quaternion=np.tile([1.,0,0,0],(4,1)),
             estimated_torque=np.tile([1.23,1.44,1.44]*4,(4,1)),joint_velocity=np.full((4,12),2.),
             gaps=np.full((4,24),.03),trunk_gaps=np.full((4,8),-.02),contact_pair_force=np.zeros((4,12)))
    m=metrics_from_arrays(a,True,seconds=.02)
    assert m['survived'] is True
    assert m['progress_world_y_m']==pytest.approx(.006)
    assert m['command_speed_mae_m_s']==pytest.approx(.1)
    np.testing.assert_allclose(m['current_rms_ideal_a'],1.)
    assert m['mean_absolute_power_W']==pytest.approx(32.88)
    assert m['minimum_trunk_proxy_gap_m']==-.02
    failed=metrics_from_arrays(a,False,seconds=.02)
    assert failed['survived'] is False


def test_pair_gate_rejects_tracking_regression_even_if_torque_improves():
    from compare_ros2 import compare_pair

    b=dict(valid=True,survived=True,progress_world_y_m=4.,command_speed_mae_m_s=.1,
           mean_joint_torque_rms_nm=10.,mean_absolute_power_W=50.,torque_peak_max_nm=40.,torque_p99_max_nm=35.,
           maximum_crossleg_contact_force_N=0.,minimum_interleg_gap_m=.01)
    c=dict(b,progress_world_y_m=3.6,mean_joint_torque_rms_nm=9.,mean_absolute_power_W=49.,command_speed_mae_m_s=.121)
    assert compare_pair(b,c)['all_gates_pass'] is False
    c['command_speed_mae_m_s']=.12
    assert compare_pair(b,c)['all_gates_pass'] is True
    c['survived']=False
    assert compare_pair(b,c)['all_gates_pass'] is False


@pytest.mark.parametrize('force,gap',[(1.01,.01),(0.,-.00201)])
def test_self_collision_blocks_selection_despite_improved_torque(force,gap):
    from compare_ros2 import compare_pair

    b=dict(valid=True,survived=True,progress_world_y_m=4.,command_speed_mae_m_s=.1,
           mean_joint_torque_rms_nm=10.,mean_absolute_power_W=50.,torque_peak_max_nm=40.,torque_p99_max_nm=35.,
           maximum_crossleg_contact_force_N=0.,minimum_interleg_gap_m=.01)
    c=dict(b,mean_joint_torque_rms_nm=9.,maximum_crossleg_contact_force_N=force,minimum_interleg_gap_m=gap)
    assert compare_pair(b,c)['all_gates_pass'] is False


def test_self_collision_gate_accepts_boundary_and_ignores_trunk_proxy():
    from compare_ros2 import compare_pair

    b=dict(valid=True,survived=True,progress_world_y_m=4.,command_speed_mae_m_s=.1,
           mean_joint_torque_rms_nm=10.,mean_absolute_power_W=50.,torque_peak_max_nm=40.,torque_p99_max_nm=35.,
           maximum_crossleg_contact_force_N=0.,minimum_interleg_gap_m=.01)
    c=dict(b,mean_joint_torque_rms_nm=9.,maximum_crossleg_contact_force_N=1.,minimum_interleg_gap_m=-.002,
           minimum_trunk_proxy_gap_m=-.006)
    result=compare_pair(b,c)
    assert result['gates']['self_collision_contact'] is True
    assert result['gates']['self_collision_geometry'] is True
    assert result['all_gates_pass'] is True
