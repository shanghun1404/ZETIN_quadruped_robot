"""V15 delay line; CPU tensors, no simulator."""

import pytest
import torch

from v15_math import TargetDelayLine


def test_each_environment_sees_its_own_delay():
    line = TargetDelayLine(torch.zeros(3, 2), max_steps=3)
    line.delay[:] = torch.tensor([0, 1, 3])
    outputs = [line.push(torch.full((3, 2), float(step))) for step in range(1, 6)]
    assert [float(o[0, 0]) for o in outputs] == [1, 2, 3, 4, 5]
    assert [float(o[1, 0]) for o in outputs] == [0, 1, 2, 3, 4]
    assert [float(o[2, 0]) for o in outputs] == [0, 0, 0, 1, 2]


def test_reset_holds_the_reset_pose_and_redraws_bounded_delays():
    line = TargetDelayLine(torch.zeros(64, 12), max_steps=3)
    for _ in range(5):
        line.push(torch.ones(64, 12))
    ids = torch.arange(0, 64, 2)
    line.reset(ids, torch.full((len(ids), 12), -1.0), generator=torch.Generator().manual_seed(0))
    assert set(line.delay[ids].tolist()) <= {0, 1, 2, 3} and len(set(line.delay[ids].tolist())) > 1
    delayed = line.push(torch.full((64, 12), 2.0))
    held = ids[line.delay[ids] > 0]
    assert bool((delayed[held] == -1.0).all())
    assert bool((delayed[ids[line.delay[ids] == 0]] == 2.0).all())
    with pytest.raises(ValueError):
        TargetDelayLine(torch.zeros(4, 12), max_steps=11)


def test_upright_scoring_and_recovery_criterion():
    from v15_math import is_recovered, standing_gate, tilt_rad, upright_score

    gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                            [0.0, 0.4226183, -0.9063078], [0.0, 0.2588190, -0.9659258]])  # up, down, side, 25/15 deg
    torch.testing.assert_close(upright_score(gravity), torch.tensor([1.0, 0.0, 0.5, 0.9531539, 0.9829629]))
    torch.testing.assert_close(standing_gate(gravity), torch.tensor([1.0, 0.0, 0.0, 0.7657695, 0.9148145]))
    torch.testing.assert_close(tilt_rad(gravity)[3:], torch.tensor([0.4363323, 0.2617994]), atol=1e-5, rtol=0)
    height = torch.tensor([0.445, 0.445, 0.2, 0.44, 0.44])
    assert is_recovered(gravity, height).tolist() == [True, False, False, False, True]
    assert is_recovered(gravity[:1], torch.tensor([0.30])).tolist() == [False]


def test_recovery_reset_orientations_cover_falls_and_upright_starts():
    from v15_math import sample_recovery_orientations, upright_score

    fallen, quat = sample_recovery_orientations(4000, 0.8, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(torch.linalg.vector_norm(quat, dim=-1), torch.ones(4000))
    assert 0.77 < float(fallen.float().mean()) < 0.83
    assert torch.equal(quat[~fallen][:, 1:3], torch.zeros(int((~fallen).sum()), 2))
    w, x, y, z = quat[fallen].unbind(-1)
    gravity_z = -(1 - 2 * (x * x + y * y))  # base-frame z of world down for unit quaternions
    scores = upright_score(torch.stack((torch.zeros_like(gravity_z),) * 2 + (gravity_z,), -1))
    assert float(scores.mean()) == pytest.approx(0.5, abs=0.03)  # uniform SO(3): mean tilt 90 deg
    with pytest.raises(ValueError):
        sample_recovery_orientations(3, 1.5)


def test_non_physical_state_flags_divergence_only():
    from v15_math import non_physical_state

    joint_vel = torch.zeros(5, 12)
    lin, ang = torch.zeros(5, 3), torch.zeros(5, 3)
    joint_vel[1, 3] = 120.0
    lin[2] = torch.tensor([8.0, 8.0, 0.0])
    ang[3, 2] = float("nan")
    joint_vel[4, 0] = 49.0  # fast but physical
    assert non_physical_state(joint_vel, lin, ang).tolist() == [False, True, True, True, False]


def test_finite_rollout_zeroes_nonfinite_rewards_and_observations():
    from v15_math import make_finite_rollout

    class FakeWrapper:
        def get_observations(self):
            return {"policy": torch.tensor([[1.0, float("nan")]])}

        def step(self, actions):
            return self.get_observations(), torch.tensor([float("inf"), 2.0]), torch.zeros(2), {}

    env = make_finite_rollout(FakeWrapper)()
    assert env.get_observations()["policy"].tolist() == [[1.0, 0.0]]
    observations, rewards, _, extras = env.step(None)
    assert rewards.tolist() == [0.0, 2.0] and observations["policy"].tolist() == [[1.0, 0.0]]
    assert extras["log"]["Diagnostics/nonfinite_steps"] == 3.0
