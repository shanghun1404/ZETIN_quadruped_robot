import pytest
import torch

from v12_vision_student import (
    TinyDepthViT,
    distillation_loss,
    normalize_depth,
    reconstruct_teacher_observation,
)


def test_depth_normalization_replaces_invalid_values_and_clips_to_range():
    depth = torch.tensor(
        [[[[0.1, 5.0, float("nan"), float("inf"), float("-inf"), 0.0, 6.0]]]]
    )

    normalized = normalize_depth(depth, near_m=0.1, far_m=5.0)

    assert torch.equal(
        normalized,
        torch.tensor([[[[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]]]]),
    )


def test_tiny_vit_maps_84_square_depth_to_the_teacher_scan_width():
    model = TinyDepthViT()

    predicted_scan = model(torch.zeros((2, 1, 84, 84)))

    assert predicted_scan.shape == (2, 187)
    assert model.patch_tokens == 144
    assert model.pos_embedding.shape == (1, 145, 128)
    assert torch.all(predicted_scan >= -1.0)
    assert torch.all(predicted_scan <= 1.0)


def test_teacher_observation_keeps_proprioception_before_predicted_scan():
    proprioception = torch.arange(2 * 48, dtype=torch.float32).reshape(2, 48)
    predicted_scan = torch.arange(2 * 187, dtype=torch.float32).reshape(2, 187)

    observation = reconstruct_teacher_observation(proprioception, predicted_scan)

    assert observation.shape == (2, 235)
    assert torch.equal(observation[:, :48], proprioception)
    assert torch.equal(observation[:, 48:], predicted_scan)


def test_teacher_observation_rejects_the_wrong_scan_width():
    with pytest.raises(ValueError, match="predicted scan must have shape"):
        reconstruct_teacher_observation(
            torch.zeros((2, 48)),
            torch.zeros((2, 186)),
        )


def test_distillation_loss_is_zero_for_exact_teacher_targets():
    scan = torch.randn((3, 187))
    actions = torch.randn((3, 12))

    losses = distillation_loss(scan, scan, actions, actions)

    assert losses["total"].item() == 0.0
    assert losses["scan"].item() == 0.0
    assert losses["action"].item() == 0.0


def test_nonzero_distillation_loss_backpropagates_finite_vit_gradients():
    torch.manual_seed(7)
    model = TinyDepthViT()
    predicted_scan = model(torch.rand((2, 1, 84, 84)))
    target_scan = torch.zeros_like(predicted_scan)
    student_actions = predicted_scan[:, :12]
    teacher_actions = torch.ones_like(student_actions)

    losses = distillation_loss(
        predicted_scan,
        target_scan,
        student_actions,
        teacher_actions,
    )
    losses["total"].backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
