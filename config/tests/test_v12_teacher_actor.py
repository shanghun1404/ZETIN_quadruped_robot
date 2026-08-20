import torch
import torch.nn.functional as F
import pytest

from v12_teacher_actor import load_frozen_v11_actor


ACTOR_DIMS = (235, 512, 256, 128, 12)
ACTOR_INDICES = (0, 2, 4, 6)


def synthetic_checkpoint(path, *, missing=None, wrong_shape=None):
    torch.manual_seed(11)
    state = {}
    for index, input_dim, output_dim in zip(
        ACTOR_INDICES,
        ACTOR_DIMS[:-1],
        ACTOR_DIMS[1:],
        strict=True,
    ):
        state[f"actor.{index}.weight"] = torch.randn(output_dim, input_dim) * 0.01
        state[f"actor.{index}.bias"] = torch.randn(output_dim) * 0.01
    if missing is not None:
        del state[missing]
    if wrong_shape is not None:
        state[wrong_shape] = torch.zeros((1, 1))
    torch.save({"model_state_dict": state}, path)
    return state


def manual_actor(observation, state):
    value = observation
    for layer, index in enumerate(ACTOR_INDICES):
        value = F.linear(
            value,
            state[f"actor.{index}.weight"],
            state[f"actor.{index}.bias"],
        )
        if layer < 3:
            value = F.elu(value)
    return value


def test_loaded_actor_exactly_matches_the_checkpoint_elu_stack(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    state = synthetic_checkpoint(checkpoint)
    observation = torch.randn((3, 235))

    actor = load_frozen_v11_actor(checkpoint)

    assert actor(observation).shape == (3, 12)
    assert torch.equal(actor(observation), manual_actor(observation, state))
    assert all(not parameter.requires_grad for parameter in actor.parameters())


def test_frozen_actor_propagates_gradients_to_its_observation(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    synthetic_checkpoint(checkpoint)
    observation = torch.randn((2, 235), requires_grad=True)
    actor = load_frozen_v11_actor(checkpoint)

    actor(observation).square().mean().backward()

    assert observation.grad is not None
    assert torch.isfinite(observation.grad).all()
    assert float(observation.grad.abs().sum()) > 0.0


def test_actor_loader_rejects_a_missing_checkpoint_key(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    synthetic_checkpoint(checkpoint, missing="actor.4.bias")

    with pytest.raises(ValueError, match="missing actor checkpoint key actor.4.bias"):
        load_frozen_v11_actor(checkpoint)


def test_actor_loader_rejects_a_wrong_checkpoint_shape(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    synthetic_checkpoint(checkpoint, wrong_shape="actor.0.weight")

    with pytest.raises(ValueError, match=r"actor.0.weight has shape \(1, 1\)"):
        load_frozen_v11_actor(checkpoint)
