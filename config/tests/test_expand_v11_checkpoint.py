import hashlib

import pytest
import torch

from expand_v11_checkpoint import expand_checkpoint, expand_input_weight


def test_input_expansion_preserves_old_columns_and_zeros_new_columns():
    old = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)

    expanded = expand_input_weight(old, new_width=7)

    assert expanded.shape == (8, 7)
    assert torch.equal(expanded[:, :3], old)
    assert torch.count_nonzero(expanded[:, 3:]) == 0


def test_input_expansion_rejects_a_smaller_width():
    with pytest.raises(ValueError, match="new width 2 is smaller than old width 3"):
        expand_input_weight(torch.zeros((8, 3)), new_width=2)


def test_checkpoint_expands_only_actor_and_critic_input_layers(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    actor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    critic = actor + 100.0
    untouched = torch.tensor([7.0, 8.0])
    actor_moment = actor + 200.0
    critic_moment = critic + 300.0
    torch.save(
        {
            "model_state_dict": {
                "std": torch.tensor([0.1, 0.2]),
                "actor.0.weight": actor,
                "actor.0.bias": untouched,
                "critic.0.weight": critic,
            },
            "optimizer_state_dict": {
                "state": {
                    0: {
                        "step": torch.tensor(99.0),
                        "exp_avg": torch.tensor([1.0, 2.0]),
                        "exp_avg_sq": torch.tensor([3.0, 4.0]),
                    },
                    1: {
                        "step": torch.tensor(99.0),
                        "exp_avg": actor_moment,
                        "exp_avg_sq": actor_moment.square(),
                    },
                    2: {
                        "step": torch.tensor(99.0),
                        "exp_avg": untouched + 10.0,
                        "exp_avg_sq": (untouched + 10.0).square(),
                    },
                    3: {
                        "step": torch.tensor(99.0),
                        "exp_avg": critic_moment,
                        "exp_avg_sq": critic_moment.square(),
                    },
                },
                "param_groups": [{"params": [0, 1, 2, 3], "lr": 2.0e-4}],
            },
            "iter": 2000,
            "infos": {"old": "kept"},
        },
        source,
    )

    expand_checkpoint(
        source,
        output,
        old_width=3,
        new_width=7,
        expected_parameter_order=(
            "std",
            "actor.0.weight",
            "actor.0.bias",
            "critic.0.weight",
        ),
    )
    result = torch.load(output, map_location="cpu", weights_only=False)

    assert torch.equal(result["model_state_dict"]["actor.0.weight"][:, :3], actor)
    assert torch.equal(result["model_state_dict"]["critic.0.weight"][:, :3], critic)
    assert torch.count_nonzero(result["model_state_dict"]["actor.0.weight"][:, 3:]) == 0
    assert torch.equal(result["model_state_dict"]["actor.0.bias"], untouched)
    optimizer = result["optimizer_state_dict"]
    assert optimizer["param_groups"][0]["lr"] == 2.0e-4
    assert torch.equal(optimizer["state"][1]["exp_avg"][:, :3], actor_moment)
    assert torch.count_nonzero(optimizer["state"][1]["exp_avg"][:, 3:]) == 0
    assert torch.equal(
        optimizer["state"][1]["exp_avg_sq"][:, :3], actor_moment.square()
    )
    assert torch.count_nonzero(optimizer["state"][1]["exp_avg_sq"][:, 3:]) == 0
    assert torch.equal(optimizer["state"][3]["exp_avg"][:, :3], critic_moment)
    assert torch.count_nonzero(optimizer["state"][3]["exp_avg"][:, 3:]) == 0
    assert torch.equal(optimizer["state"][2]["exp_avg"], untouched + 10.0)
    assert result["iter"] == 2000
    assert result["infos"]["old"] == "kept"
    bootstrap = result["infos"]["bootstrap"]
    assert bootstrap["old_width"] == 3
    assert bootstrap["new_width"] == 7
    assert bootstrap["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_checkpoint_rejects_an_unexpected_source_width(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    torch.save(
        {
            "model_state_dict": {
                "actor.0.weight": torch.zeros((4, 4)),
                "critic.0.weight": torch.zeros((4, 4)),
            },
            "optimizer_state_dict": {},
            "iter": 0,
            "infos": {},
        },
        source,
    )

    with pytest.raises(ValueError, match="actor.0.weight has width 4, expected 3"):
        expand_checkpoint(
            source,
            output,
            old_width=3,
            new_width=7,
            expected_parameter_order=("actor.0.weight", "critic.0.weight"),
        )


def _minimal_checkpoint(parameter_order, missing_adam_field=None, amsgrad=False):
    weights = {
        "std": torch.ones(2),
        "actor.0.weight": torch.ones((4, 3)),
        "actor.0.bias": torch.ones(4),
        "critic.0.weight": torch.ones((4, 3)),
    }
    state = {}
    for parameter_id, name in enumerate(parameter_order):
        adam_state = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.ones_like(weights[name]),
            "exp_avg_sq": torch.ones_like(weights[name]),
        }
        if name == "actor.0.weight" and missing_adam_field in adam_state:
            adam_state.pop(missing_adam_field)
        if amsgrad:
            adam_state["max_exp_avg_sq"] = torch.ones_like(weights[name])
        state[parameter_id] = adam_state
    if missing_adam_field == "max_exp_avg_sq":
        state[parameter_order.index("actor.0.weight")].pop("max_exp_avg_sq")
    return {
        "model_state_dict": {name: weights[name] for name in parameter_order},
        "optimizer_state_dict": {
            "state": state,
            "param_groups": [
                {"params": list(range(len(parameter_order))), "amsgrad": amsgrad}
            ],
        },
        "iter": 1,
        "infos": {},
    }


def test_checkpoint_rejects_unexpected_model_parameter_order(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    expected = (
        "std",
        "actor.0.weight",
        "actor.0.bias",
        "critic.0.weight",
    )
    reordered = (expected[1], expected[0], *expected[2:])
    torch.save(_minimal_checkpoint(reordered), source)

    with pytest.raises(ValueError, match="model parameter order"):
        expand_checkpoint(
            source,
            output,
            old_width=3,
            new_width=7,
            expected_parameter_order=expected,
        )


@pytest.mark.parametrize("missing_field", ("step", "exp_avg", "exp_avg_sq"))
def test_checkpoint_rejects_incomplete_adam_state(tmp_path, missing_field):
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    expected = (
        "std",
        "actor.0.weight",
        "actor.0.bias",
        "critic.0.weight",
    )
    torch.save(_minimal_checkpoint(expected, missing_field), source)

    with pytest.raises(ValueError, match=f"missing Adam state {missing_field}"):
        expand_checkpoint(
            source,
            output,
            old_width=3,
            new_width=7,
            expected_parameter_order=expected,
        )


def test_checkpoint_rejects_missing_amsgrad_maximum(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    expected = (
        "std",
        "actor.0.weight",
        "actor.0.bias",
        "critic.0.weight",
    )
    torch.save(
        _minimal_checkpoint(expected, "max_exp_avg_sq", amsgrad=True),
        source,
    )

    with pytest.raises(ValueError, match="missing Adam state max_exp_avg_sq"):
        expand_checkpoint(
            source,
            output,
            old_width=3,
            new_width=7,
            expected_parameter_order=expected,
        )
