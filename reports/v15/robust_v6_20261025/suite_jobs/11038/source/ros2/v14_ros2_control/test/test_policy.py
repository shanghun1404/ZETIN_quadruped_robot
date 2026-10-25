"""ROS2 policy observation layout, actor loading and depth path; CPU, no ROS."""

import numpy as np
import pytest
import torch
from torch import nn

from v12_vision_student import TinyDepthViT, normalize_depth
from v14_ros2_control import contract
from v14_ros2_control.policy import V14DepthPolicy, load_actor, projected_gravity


def rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def test_projected_gravity_is_world_down_in_the_base_frame():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        expected = rotation_matrix(q).T @ np.array([0.0, 0.0, -1.0])
        actual = projected_gravity(torch.tensor(q)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def rsl_actor_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    reference = nn.Sequential(nn.Linear(235, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                              nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))
    state = {f"actor.{k}": v for k, v in reference.state_dict().items()}
    state.update({"std": torch.ones(12), "critic.0.weight": torch.zeros(1, 235), "critic.0.bias": torch.zeros(1)})
    torch.save({"model_state_dict": state, "iter": 5700}, path)
    return reference


def student_checkpoint(path, seed=1):
    torch.manual_seed(seed)
    model = TinyDepthViT()
    torch.save({"model_state_dict": model.state_dict()}, path)
    return model.eval()


def test_actor_is_rebuilt_exactly_from_an_rsl_checkpoint(tmp_path):
    reference = rsl_actor_checkpoint(tmp_path / "actor.pt")
    actor = load_actor(tmp_path / "actor.pt")
    observation = torch.randn(9, 235)
    torch.testing.assert_close(actor(observation), reference(observation), rtol=0, atol=0)
    torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(12, 48), "actor.0.bias": torch.zeros(12)}},
               tmp_path / "wrong.pt")
    with pytest.raises(ValueError):
        load_actor(tmp_path / "wrong.pt")


def test_observation_matches_the_v11_v12_policy_layout(tmp_path):
    actor = rsl_actor_checkpoint(tmp_path / "actor.pt")
    student = student_checkpoint(tmp_path / "student.pt")
    policy = V14DepthPolicy(tmp_path / "actor.pt", tmp_path / "student.pt")
    seen = []
    original = policy.actor
    policy.actor = lambda observation: seen.append(observation.clone()) or original(observation)

    rng = np.random.default_rng(5)
    depth = rng.uniform(0.0, 6.0, size=(84, 84)).astype(np.float32)
    depth[0, 0] = np.inf
    inputs = dict(
        joint_pos=rng.normal(size=12), joint_vel=rng.normal(size=12),
        lin_vel_b=(0.1, 0.4, -0.02), ang_vel_b=(0.01, -0.03, 0.2), quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        command=(0.0, 0.4, 0.0), depth_m=depth,
    )
    first_action, first_target = policy.step(**inputs)
    second_action, _ = policy.step(**inputs)

    default = torch.tensor([contract.joint_default(n) for n in contract.JOINT_NAMES])
    obs = seen[1][0]
    torch.testing.assert_close(obs[0:3], torch.tensor(inputs["lin_vel_b"]).float())
    torch.testing.assert_close(obs[3:6], torch.tensor(inputs["ang_vel_b"]).float())
    torch.testing.assert_close(obs[6:9], torch.tensor([0.0, 0.0, -1.0]))
    torch.testing.assert_close(obs[9:12], torch.tensor(inputs["command"]).float())
    torch.testing.assert_close(obs[12:24], torch.tensor(inputs["joint_pos"]).float() - default)
    torch.testing.assert_close(obs[24:36], torch.tensor(inputs["joint_vel"]).float())
    torch.testing.assert_close(obs[36:48], first_action)  # last action is the raw policy output
    assert torch.equal(seen[0][0, 36:48], torch.zeros(12))
    with torch.no_grad():
        expected_scan = student(normalize_depth(torch.tensor(depth)[None, None]))[0]
    torch.testing.assert_close(obs[48:], expected_scan)
    torch.testing.assert_close(first_target, default + 0.2 * first_action)
    with torch.no_grad():
        torch.testing.assert_close(second_action, actor(seen[1])[0])
    with pytest.raises(ValueError):
        policy.step(**{**inputs, "depth_m": depth[:80]})
