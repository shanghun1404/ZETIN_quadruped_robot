import argparse

from evaluation_cli import (
    add_fixed_terrain_level_argument,
    add_seed_argument,
    evaluation_metadata,
    load_inference_checkpoint,
    validate_fixed_terrain_level,
)


def test_seed_argument_defaults_to_historical_seed_42():
    parser = argparse.ArgumentParser()
    add_seed_argument(parser)

    args = parser.parse_args([])

    assert args.seed == 42


def test_seed_argument_accepts_an_explicit_seed():
    parser = argparse.ArgumentParser()
    add_seed_argument(parser)

    args = parser.parse_args(["--seed", "46"])

    assert args.seed == 46


def test_evaluation_metadata_identifies_seed_and_checkpoint():
    result = evaluation_metadata("/runs/model_2000.pt", seed=45)

    assert result == {"checkpoint": "/runs/model_2000.pt", "seed": 45}


def test_inference_checkpoint_load_skips_optimizer_state():
    class RecordingRunner:
        def __init__(self):
            self.call = None

        def load(self, path, **kwargs):
            self.call = (path, kwargs)

    runner = RecordingRunner()

    load_inference_checkpoint(runner, "/runs/expanded_model.pt")

    assert runner.call == (
        "/runs/expanded_model.pt",
        {"load_optimizer": False},
    )


def test_fixed_terrain_level_is_optional_and_parses_an_integer():
    parser = argparse.ArgumentParser()
    add_fixed_terrain_level_argument(parser)

    assert parser.parse_args([]).terrain_level is None
    assert parser.parse_args(["--terrain-level", "4"]).terrain_level == 4


def test_fixed_terrain_level_must_exist_in_generator_rows():
    assert validate_fixed_terrain_level(4, num_rows=10) == 4

    try:
        validate_fixed_terrain_level(10, num_rows=10)
    except ValueError as error:
        assert "0..9" in str(error)
    else:
        raise AssertionError("out-of-range terrain level was accepted")
