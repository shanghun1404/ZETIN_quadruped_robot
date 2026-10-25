"""Verify that zero scanner inputs preserve source actor and critic outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional


def _forward(state: dict[str, torch.Tensor], prefix: str, value: torch.Tensor) -> torch.Tensor:
    for layer in (0, 2, 4):
        value = functional.elu(
            functional.linear(
                value,
                state[f"{prefix}.{layer}.weight"],
                state[f"{prefix}.{layer}.bias"],
            )
        )
    return functional.linear(
        value,
        state[f"{prefix}.6.weight"],
        state[f"{prefix}.6.bias"],
    )


def _optimizer_parameter_state(checkpoint: dict, parameter_name: str) -> dict:
    optimizer = checkpoint["optimizer_state_dict"]
    parameter_ids = [
        parameter_id
        for group in optimizer["param_groups"]
        for parameter_id in group["params"]
    ]
    parameter_index = list(checkpoint["model_state_dict"]).index(parameter_name)
    return optimizer["state"][parameter_ids[parameter_index]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("expanded")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    expanded = torch.load(args.expanded, map_location="cpu", weights_only=False)
    source_state = source["model_state_dict"]
    expanded_state = expanded["model_state_dict"]
    generator = torch.Generator().manual_seed(20260812)
    old_input = torch.randn((128, 48), generator=generator)
    new_input = torch.cat((old_input, torch.zeros((128, 187))), dim=1)
    actor_error = float(
        (_forward(source_state, "actor", old_input) - _forward(expanded_state, "actor", new_input))
        .abs()
        .max()
    )
    critic_error = float(
        (_forward(source_state, "critic", old_input) - _forward(expanded_state, "critic", new_input))
        .abs()
        .max()
    )
    optimizer_checks = {}
    for name in ("actor.0.weight", "critic.0.weight"):
        source_parameter = _optimizer_parameter_state(source, name)
        expanded_parameter = _optimizer_parameter_state(expanded, name)
        for moment_name in ("exp_avg", "exp_avg_sq"):
            source_moment = source_parameter[moment_name]
            expanded_moment = expanded_parameter[moment_name]
            optimizer_checks[f"{name}.{moment_name}.old_max_abs_error"] = float(
                (source_moment - expanded_moment[:, :48]).abs().max()
            )
            optimizer_checks[f"{name}.{moment_name}.new_nonzero_count"] = int(
                torch.count_nonzero(expanded_moment[:, 48:])
            )
    report = {
        "actor_max_abs_error": actor_error,
        "critic_max_abs_error": critic_error,
        "source_actor_shape": list(source_state["actor.0.weight"].shape),
        "expanded_actor_shape": list(expanded_state["actor.0.weight"].shape),
        "new_actor_nonzero_count": int(
            torch.count_nonzero(expanded_state["actor.0.weight"][:, 48:])
        ),
        "new_critic_nonzero_count": int(
            torch.count_nonzero(expanded_state["critic.0.weight"][:, 48:])
        ),
        "expanded_iteration": int(expanded["iter"]),
        "source_iteration": int(source["iter"]),
        "source_optimizer_lr": float(
            source["optimizer_state_dict"]["param_groups"][0]["lr"]
        ),
        "expanded_optimizer_lr": float(
            expanded["optimizer_state_dict"]["param_groups"][0]["lr"]
        ),
        "optimizer_checks": optimizer_checks,
    }
    report["ok"] = (
        actor_error <= 1e-6
        and critic_error <= 1e-6
        and report["new_actor_nonzero_count"] == 0
        and report["new_critic_nonzero_count"] == 0
        and report["expanded_iteration"] == report["source_iteration"]
        and report["expanded_optimizer_lr"] == report["source_optimizer_lr"]
        and all(
            value == 0
            for key, value in optimizer_checks.items()
            if key.endswith(("old_max_abs_error", "new_nonzero_count"))
        )
    )
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
