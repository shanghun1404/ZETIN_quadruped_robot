import pytest

from training_resume import (
    load_training_checkpoint,
    validate_initial_terrain_level,
)


def test_training_resume_loads_optimizer_and_restores_adaptive_learning_rate():
    class Algorithm:
        def __init__(self):
            self.optimizer = type("Optimizer", (), {"param_groups": [{"lr": 1.0e-3}]})()
            self.learning_rate = 1.0e-3

    class RecordingRunner:
        def __init__(self):
            self.alg = Algorithm()
            self.call = None

        def load(self, path, **kwargs):
            self.call = (path, kwargs)
            self.alg.optimizer.param_groups[0]["lr"] = 1.1390625e-4

    runner = RecordingRunner()

    load_training_checkpoint(runner, "/runs/expanded.pt", map_location="cuda:0")

    assert runner.call == (
        "/runs/expanded.pt",
        {"load_optimizer": True, "map_location": "cuda:0"},
    )
    assert runner.alg.learning_rate == 1.1390625e-4


def test_resume_terrain_level_must_exist_in_generator_rows():
    assert validate_initial_terrain_level(5, num_rows=10) == 5

    with pytest.raises(ValueError, match="0..9"):
        validate_initial_terrain_level(10, num_rows=10)
