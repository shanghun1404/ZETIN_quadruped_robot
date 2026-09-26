import math

import torch

from v14_heading_math import lane_yaw_rate, update_lane_reference


def test_plus_y_robot_corrects_heading_and_right_side_error():
    z = torch.zeros(3)
    yaw = torch.tensor([0.2, -0.2, 0.])
    lateral = torch.tensor([0., 0., 0.5])
    actual = lane_yaw_rate(yaw, z, lateral)
    assert torch.allclose(actual, torch.tensor([-0.3, 0.3, 0.3]))


def test_heading_wrap_saturation_and_standing():
    actual = lane_yaw_rate(torch.tensor([math.pi-.05, -2., 1.]),
                          torch.tensor([-math.pi+.05, 0., 0.]), torch.zeros(3),
                          standing=torch.tensor([False, False, True]))
    assert torch.allclose(actual, torch.tensor([.15, .4, 0.]), atol=1e-6)


def test_reference_updates_on_episode_reset_not_command_resample():
    xy, yaw = torch.zeros((3, 2)), torch.zeros(3)
    root_xy, root_yaw = torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), torch.ones(3)
    update_lane_reference(xy, yaw, root_xy, root_yaw,
                          torch.tensor([0, 1, 0]), torch.tensor([0, 1]))
    assert torch.equal(xy, torch.tensor([[1., 2.], [0., 0.], [0., 0.]]))
    assert torch.equal(yaw, torch.tensor([1., 0., 0.]))
