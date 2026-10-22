"""CPU-only checks; importing the overlay must never start IsaacLab."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "config/experiments/v16_20260929/v15_torque_finetune.py"


def overlay():
    assert MODULE.exists(), "Finetune implementation is not present yet"
    spec = importlib.util.spec_from_file_location("finetune_contract", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinetuneContractTests(unittest.TestCase):
    def test_lr_reset_keeps_optimizer_moments_and_syncs_all_groups(self):
        module = overlay()
        state = {0: {"step": 9, "exp_avg": [1., 2.]}}
        optimizer = SimpleNamespace(param_groups=[{"lr": .001}, {"lr": .0001}], state=state)
        algorithm = SimpleNamespace(optimizer=optimizer, learning_rate=.001, schedule="adaptive")
        result = module.synchronize_fixed_lr(algorithm)
        self.assertEqual(algorithm.learning_rate, 3.375e-5)
        self.assertEqual(algorithm.schedule, "fixed")
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [3.375e-5, 3.375e-5])
        self.assertIs(optimizer.state, state)
        self.assertEqual(state[0]["exp_avg"], [1., 2.])
        self.assertEqual(result["optimizer_group_lr_before"], [.001, .0001])
        self.assertTrue(result["algorithm_optimizer_lr_match"])

    def test_missing_optimizer_groups_are_rejected(self):
        module = overlay()
        algorithm = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[], state={}), learning_rate=.001, schedule="adaptive")
        with self.assertRaises(ValueError):
            module.synchronize_fixed_lr(algorithm)

    def test_changed_original_is_rejected_before_compilation(self):
        module = overlay()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unreviewed.py"
            path.write_text("raise RuntimeError('must never execute')\n")
            with self.assertRaises(RuntimeError):
                module.build_source(path)

    def test_construction_parses_without_importing_physics(self):
        module = overlay()
        source = module.build_source(ROOT / "config/train_v15_locomotion.py")
        ast.parse(source)
        self.assertIn("make_finite_rollout(make_v13_depth_wrapper())", source)
        self.assertIn("env_cfg.actions.joint_pos.cutoff_hz = 23.0", source)
        self.assertIn("env_cfg.actions.joint_pos.effort_guard_fraction = 0.97", source)
        self.assertIn('"episode_length_s": float(env_cfg.episode_length_s)', source)


if __name__ == "__main__":
    unittest.main()
