"""Exact frozen reconstruction of the selected V11 RSL-RL actor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


ACTOR_LAYER_SPECS = (
    ("actor.0", 235, 512),
    ("actor.2", 512, 256),
    ("actor.4", 256, 128),
    ("actor.6", 128, 12),
)


class FrozenV11Actor(nn.Module):
    """The V11 deterministic actor mean with immutable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(235, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != 235:
            raise ValueError("teacher observation must have shape [batch, 235]")
        return self.actor(observation)


def load_frozen_v11_actor(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> FrozenV11Actor:
    """Load only the validated V11 actor layers and freeze their parameters."""

    checkpoint_data = torch.load(
        Path(checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint_data.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")

    model = FrozenV11Actor()
    linear_layers = (model.actor[0], model.actor[2], model.actor[4], model.actor[6])
    with torch.no_grad():
        for (prefix, input_dim, output_dim), layer in zip(
            ACTOR_LAYER_SPECS,
            linear_layers,
            strict=True,
        ):
            for suffix, expected_shape in (
                ("weight", (output_dim, input_dim)),
                ("bias", (output_dim,)),
            ):
                key = f"{prefix}.{suffix}"
                if key not in state:
                    raise ValueError(f"missing actor checkpoint key {key}")
                tensor = state[key]
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"{key} has shape {tuple(tensor.shape)}, expected {expected_shape}"
                    )
                getattr(layer, suffix).copy_(tensor)

    model.requires_grad_(False)
    model.eval()
    return model.to(device)
