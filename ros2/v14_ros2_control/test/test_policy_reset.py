import torch
from v14_ros2_control.policy import V14DepthPolicy


def test_reset_after_inference_can_be_repeated_and_mutated():
    policy=object.__new__(V14DepthPolicy)
    for _ in range(3):
        with torch.inference_mode():
            policy.last_action=torch.ones(12)
        policy.reset()
        torch.testing.assert_close(policy.last_action,torch.zeros(12))
        policy.last_action.add_(1.)
