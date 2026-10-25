"""Run a deterministic synthetic smoke check for the V12 vision front-end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from v12_vision_student import (
    TinyDepthViT,
    distillation_loss,
    normalize_depth,
    reconstruct_teacher_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = TinyDepthViT()
    metric_depth = torch.rand((4, 1, 84, 84)) * 6.0
    metric_depth[0, 0, 0, 0] = float("nan")
    metric_depth[0, 0, 0, 1] = float("inf")
    normalized_depth = normalize_depth(metric_depth)

    predicted_scan = model(normalized_depth)
    proprioception = torch.randn((4, 48))
    teacher_observation = reconstruct_teacher_observation(
        proprioception,
        predicted_scan,
    )
    target_scan = torch.tanh(torch.randn_like(predicted_scan))
    student_actions = predicted_scan[:, :12]
    teacher_actions = torch.tanh(torch.randn_like(student_actions))
    losses = distillation_loss(
        predicted_scan,
        target_scan,
        student_actions,
        teacher_actions,
    )
    losses["total"].backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    finite_gradients = all(
        gradient is not None and torch.isfinite(gradient).all().item()
        for gradient in gradients
    )
    gradient_l1 = sum(
        float(gradient.detach().abs().sum())
        for gradient in gradients
        if gradient is not None
    )
    all_finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in (normalized_depth, predicted_scan, teacher_observation)
    )

    checkpoint_path = args.output_dir / "tiny_depth_vit_initial.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": {
                "input": [1, 84, 84],
                "patch_size": 7,
                "patch_tokens": model.patch_tokens,
                "embed_dim": 128,
                "transformer_blocks": 4,
                "attention_heads": 4,
                "feedforward_dim": 256,
                "output_scan_dim": 187,
            },
            "seed": args.seed,
            "status": "initialized_only_not_trained",
        },
        checkpoint_path,
    )

    report = {
        "status": "synthetic_smoke_passed_not_trained",
        "seed": args.seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(normalized_depth.shape),
        "predicted_scan_shape": list(predicted_scan.shape),
        "teacher_observation_shape": list(teacher_observation.shape),
        "patch_tokens": model.patch_tokens,
        "loss": {name: float(value.detach()) for name, value in losses.items()},
        "finite_forward_tensors": all_finite,
        "finite_model_gradients": finite_gradients,
        "gradient_l1": gradient_l1,
        "teacher_actor_connected": False,
        "checkpoint": checkpoint_path.name,
    }
    if not all_finite or not finite_gradients or gradient_l1 <= 0.0:
        raise RuntimeError(f"V12 vision smoke check failed: {report}")

    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
