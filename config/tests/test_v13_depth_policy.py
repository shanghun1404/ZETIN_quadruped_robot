import pytest
import torch

from v13_depth_policy import compose_depth_policy_observation


def test_depth_scan_replaces_teacher_scan_but_preserves_proprioception() -> None:
    teacher_observation = torch.arange(235, dtype=torch.float32).reshape(1, 235)
    predicted_scan = torch.full((1, 187), 0.25)
    new_observation = compose_depth_policy_observation(teacher_observation, predicted_scan)
    assert new_observation.shape == (1, 235)
    assert torch.equal(new_observation[:, :48], teacher_observation[:, :48])
    assert torch.equal(new_observation[:, 48:], predicted_scan)
    assert not torch.equal(new_observation[:, 48:], teacher_observation[:, 48:])


def test_depth_scan_requires_v12_shapes() -> None:
    with pytest.raises(ValueError, match="235"):
        compose_depth_policy_observation(torch.zeros(1, 234), torch.zeros(1, 187))
