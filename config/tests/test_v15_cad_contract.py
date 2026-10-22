import numpy as np
import json
import pytest
from v15_cad_contract import joint_mapping, policy_values, physical_values

POLICY=('FL_HAA','FR_HAA','FL_HFE','FR_HFE','RR_HAA','RL_HAA',
        'RR_HFE','RL_HFE','FL_KFE','FR_KFE','RR_KFE','RL_KFE')
PHYSICAL=('FL_hip_roll','FR_hip_roll','FL_hip_pitch','FR_hip_pitch',
          'BR_hip_roll','BL_hip_roll','BR_hip_pitch','BL_hip_pitch',
          'FL_knee','FR_knee','BR_knee','BL_knee')

def test_mirrored_joint_axes_produce_same_physical_rotation_and_work():
    names,sign=joint_mapping(POLICY)
    assert names==PHYSICAL
    # Measured USD world axes: canonical HAA +Y, HFE/KFE +X.
    raw=np.array([(0,-1,0),(0,-1,0),(-1,0,0),(1,0,0),
                  (0,1,0),(0,1,0),(1,0,0),(-1,0,0),
                  (1,0,0),(-1,0,0),(-1,0,0),(1,0,0)])
    expected=np.array([(0,1,0) if n.endswith('HAA') else (1,0,0) for n in POLICY])
    np.testing.assert_allclose(raw*sign[:,None],expected)
    q=np.linspace(-.4,1.2,12);tau=np.linspace(-22,40,12)
    np.testing.assert_array_equal(policy_values(physical_values(q,sign),sign),q)
    assert np.dot(physical_values(q,sign),physical_values(tau,sign))==pytest.approx(np.dot(q,tau))

def test_name_mapping_rejects_unknown_or_duplicate_joints():
    with pytest.raises(ValueError):joint_mapping(('FL_HAA','FL_HAA'))
    with pytest.raises(ValueError):joint_mapping(('FL_ANKLE',))
    _,sign=joint_mapping(POLICY)
    with pytest.raises(ValueError):physical_values(np.zeros(11),sign)

def test_physical_command_fields_keep_native_ros_numeric_and_json_types():
    from v15_cad_contract import physical_command_fields
    _,sign=joint_mapping(POLICY)
    q=np.linspace(-.5,1.1,12,dtype=np.float64)
    effort=np.linspace(-10.,20.,12,dtype=np.float64)
    position,torque=physical_command_fields(q,effort,sign)
    # Native float lists preserve the reused bridge's float32 Torch construction.
    assert isinstance(position,list) and isinstance(torque,list)
    assert all(type(v) is float for v in position+torque)
    decoded=json.loads(json.dumps(dict(position=position,feedforward=torque)))
    np.testing.assert_allclose(decoded['position'],q*sign)
    np.testing.assert_allclose(decoded['feedforward'],effort*sign)
