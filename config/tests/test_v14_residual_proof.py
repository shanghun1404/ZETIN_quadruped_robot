"""Zero-action proof must detect state changes and ignored termination differences."""

import numpy as np
import pytest

from v14_residual_proof import compare_rollouts


def traces():
    return {'valid':np.array([[True,True],[True,False]]), 'alive_at_end':np.array([True,False]),
            'positions':np.zeros((2,2,3)), 'quaternions':np.zeros((2,2,4)),
            'ground_height':np.zeros((2,2)), 'commands':np.zeros((2,2,3)),
            'linear_velocity_w':np.zeros((2,2,3)), 'angular_velocity_b':np.zeros((2,2,3)),
            'time_s':np.array([0.,.02])}


def test_zero_rollout_proof_ignores_post_reset_but_detects_live_motion():
    base, other = traces(), traces()
    other['positions'][1,1] = 100  # Reset robot has already ended its measured episode.
    assert compare_rollouts(base,other)['passed']
    other['positions'][1,0,0] = .01
    assert not compare_rollouts(base,other)['passed']


def test_zero_rollout_proof_detects_survival_and_timing_changes():
    base, other = traces(), traces()
    other['alive_at_end'][1] = True
    assert not compare_rollouts(base,other)['passed']
    other = traces(); other['valid'][1,1] = True
    assert not compare_rollouts(base,other)['passed']
    other = traces(); other['time_s'][1] = .04
    assert not compare_rollouts(base,other)['passed']
