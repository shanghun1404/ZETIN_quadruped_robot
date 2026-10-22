"""Tiny depth ViT front-end for distilling the V11 height-scan teacher.

The module deliberately stops at reconstructing the existing 235-dimensional
teacher observation.  A future data-collection run can train the vision front-
end while keeping the already validated V11 actor frozen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


DEPTH_HEIGHT = 84
DEPTH_WIDTH = 84
PROPRIOCEPTION_DIM = 48
TEACHER_SCAN_DIM = 187


def normalize_depth(
    depth: torch.Tensor,
    *,
    near_m: float = 0.1,
    far_m: float = 5.0,
) -> torch.Tensor:
    """Clip metric depth and map it to [0, 1], treating invalid pixels as far."""

    if not isinstance(depth, torch.Tensor) or not depth.is_floating_point():
        raise TypeError("depth must be a floating-point torch.Tensor")
    if far_m <= near_m:
        raise ValueError("far_m must be greater than near_m")

    cleaned = torch.nan_to_num(depth, nan=far_m, posinf=far_m, neginf=far_m)
    clipped = cleaned.clamp(min=near_m, max=far_m)
    return (clipped - near_m) / (far_m - near_m)


class TinyDepthViT(nn.Module):
    """Map one 84x84 normalized depth frame to the V11 187-value scan."""

    def __init__(
        self,
        *,
        image_size: int = DEPTH_HEIGHT,
        patch_size: int = 7,
        in_channels: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: int = 2,
        scan_dim: int = TEACHER_SCAN_DIM,
    ) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if embed_dim <= 0 or heads <= 0 or embed_dim % heads:
            raise ValueError("embed_dim must be positive and divisible by heads")
        if depth <= 0 or mlp_ratio <= 0 or scan_dim <= 0:
            raise ValueError("depth, mlp_ratio, and scan_dim must be positive")

        self.image_size = image_size
        self.in_channels = in_channels
        self.scan_dim = scan_dim
        self.patch_tokens = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.patch_tokens + 1, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.scan_head = nn.Linear(embed_dim, scan_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, normalized_depth: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, self.image_size, self.image_size)
        if normalized_depth.ndim != 4 or tuple(normalized_depth.shape[1:]) != expected:
            raise ValueError(
                "normalized depth must have shape "
                f"[batch, {self.in_channels}, {self.image_size}, {self.image_size}]"
            )

        patches = self.patch_embedding(normalized_depth).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(normalized_depth.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1) + self.pos_embedding
        encoded = self.encoder(tokens)
        return torch.tanh(self.scan_head(self.norm(encoded[:, 0])))


def reconstruct_teacher_observation(
    proprioception: torch.Tensor,
    predicted_scan: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the unchanged 48-value proprioception and predicted scan."""

    if proprioception.ndim != 2 or proprioception.shape[1] != PROPRIOCEPTION_DIM:
        raise ValueError(
            f"proprioception must have shape [batch, {PROPRIOCEPTION_DIM}]"
        )
    if predicted_scan.ndim != 2 or predicted_scan.shape[1] != TEACHER_SCAN_DIM:
        raise ValueError(f"predicted scan must have shape [batch, {TEACHER_SCAN_DIM}]")
    if proprioception.shape[0] != predicted_scan.shape[0]:
        raise ValueError("proprioception and predicted scan batch sizes must match")
    if proprioception.device != predicted_scan.device:
        raise ValueError("proprioception and predicted scan must be on the same device")
    return torch.cat((proprioception, predicted_scan), dim=1)


def distillation_loss(
    predicted_scan: torch.Tensor,
    target_scan: torch.Tensor,
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    action_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Combine pseudo-scan regression with frozen-teacher action matching."""

    if predicted_scan.shape != target_scan.shape:
        raise ValueError("predicted and target scan shapes must match")
    if predicted_scan.ndim != 2 or predicted_scan.shape[1] != TEACHER_SCAN_DIM:
        raise ValueError(f"scan tensors must have shape [batch, {TEACHER_SCAN_DIM}]")
    if student_actions.shape != teacher_actions.shape:
        raise ValueError("student and teacher action shapes must match")
    if student_actions.ndim != 2 or student_actions.shape[0] != predicted_scan.shape[0]:
        raise ValueError("action tensors must be two-dimensional and share the scan batch")
    if action_weight < 0:
        raise ValueError("action_weight must be non-negative")

    scan_loss = F.smooth_l1_loss(predicted_scan, target_scan)
    action_loss = F.mse_loss(student_actions, teacher_actions)
    total_loss = scan_loss + action_weight * action_loss
    return {"total": total_loss, "scan": scan_loss, "action": action_loss}
