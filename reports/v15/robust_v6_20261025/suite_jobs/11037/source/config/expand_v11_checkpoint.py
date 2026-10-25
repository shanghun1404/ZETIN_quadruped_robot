"""Expand a 48-input locomotion checkpoint for a 187-point terrain scan."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


V10_ACTOR_CRITIC_PARAMETER_ORDER = (
    "std",
    "actor.0.weight",
    "actor.0.bias",
    "actor.2.weight",
    "actor.2.bias",
    "actor.4.weight",
    "actor.4.bias",
    "actor.6.weight",
    "actor.6.bias",
    "critic.0.weight",
    "critic.0.bias",
    "critic.2.weight",
    "critic.2.bias",
    "critic.4.weight",
    "critic.4.bias",
    "critic.6.weight",
    "critic.6.bias",
)


def expand_input_weight(weight: torch.Tensor, new_width: int) -> torch.Tensor:
    """Copy an input layer into a wider zero-initialized layer."""
    old_width = int(weight.shape[1])
    if new_width < old_width:
        raise ValueError(
            f"new width {new_width} is smaller than old width {old_width}"
        )
    expanded = weight.new_zeros((weight.shape[0], new_width))
    expanded[:, :old_width] = weight
    return expanded


def expand_optimizer_input_moments(
    checkpoint: dict,
    parameter_names: tuple[str, ...],
    new_width: int,
) -> None:
    """Expand Adam moments for widened parameters without losing resume state."""
    optimizer = checkpoint["optimizer_state_dict"]
    parameter_ids = []
    parameter_groups = {}
    for group in optimizer["param_groups"]:
        for parameter_id in group["params"]:
            if parameter_id in parameter_groups:
                raise ValueError(f"optimizer parameter {parameter_id} appears twice")
            parameter_ids.append(parameter_id)
            parameter_groups[parameter_id] = group
    model_names = list(checkpoint["model_state_dict"])
    if len(parameter_ids) != len(model_names):
        raise ValueError(
            "optimizer parameter count does not match model state order: "
            f"{len(parameter_ids)} != {len(model_names)}"
        )

    for name in parameter_names:
        parameter_id = parameter_ids[model_names.index(name)]
        parameter_state = optimizer["state"].get(parameter_id)
        if not isinstance(parameter_state, dict):
            raise ValueError(f"missing Adam state for {name}")
        required_fields = ["step", "exp_avg", "exp_avg_sq"]
        if parameter_groups[parameter_id].get("amsgrad", False):
            required_fields.append("max_exp_avg_sq")
        for field in required_fields:
            if field not in parameter_state:
                raise ValueError(f"missing Adam state {field} for {name}")
        for moment_name in required_fields[1:]:
            moment = parameter_state[moment_name]
            if tuple(moment.shape) != tuple(
                checkpoint["model_state_dict"][name].shape
            ):
                raise ValueError(
                    f"optimizer {moment_name} for {name} has shape "
                    f"{tuple(moment.shape)}, expected "
                    f"{tuple(checkpoint['model_state_dict'][name].shape)}"
                )
            parameter_state[moment_name] = expand_input_weight(moment, new_width)


def expand_checkpoint(
    source: str | Path,
    output: str | Path,
    old_width: int = 48,
    new_width: int = 235,
    expected_parameter_order: tuple[str, ...] = V10_ACTOR_CRITIC_PARAMETER_ORDER,
) -> None:
    """Expand actor/critic inputs and their optimizer state for exact resume."""
    source_path = Path(source)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    actual_parameter_order = tuple(state)
    if actual_parameter_order != expected_parameter_order:
        raise ValueError(
            "model parameter order does not match the validated V10 architecture: "
            f"{actual_parameter_order}"
        )
    widened_parameters = ("actor.0.weight", "critic.0.weight")
    for key in widened_parameters:
        width = int(state[key].shape[1])
        if width != old_width:
            raise ValueError(f"{key} has width {width}, expected {old_width}")
    expand_optimizer_input_moments(
        checkpoint,
        widened_parameters,
        new_width=new_width,
    )
    for key in widened_parameters:
        state[key] = expand_input_weight(state[key], new_width)

    infos = dict(checkpoint.get("infos") or {})
    infos["bootstrap"] = {
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "old_width": old_width,
        "new_width": new_width,
    }
    checkpoint["infos"] = infos
    torch.save(checkpoint, Path(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--old-width", type=int, default=48)
    parser.add_argument("--new-width", type=int, default=235)
    args = parser.parse_args()
    expand_checkpoint(args.source, args.output, args.old_width, args.new_width)


if __name__ == "__main__":
    main()
