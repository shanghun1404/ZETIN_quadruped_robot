"""CPU contracts for the separately prepared torque reward pilot."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "config/experiments/v16_20260929/v15_torque_reward_finetune.py"


def overlay():
    spec = importlib.util.spec_from_file_location("reward_finetune_contract", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewards(weight=-.0005):
    function = SimpleNamespace(__module__="isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.rewards",
                               __name__="joint_torques_penalty")
    return SimpleNamespace(joint_torques=SimpleNamespace(weight=weight, func=function),
                           gait=SimpleNamespace(weight=10.), joint_power=None)


class RewardFinetuneTests(unittest.TestCase):
    def test_only_existing_torque_norm_weight_changes(self):
        module = overlay()
        cfg = rewards()
        result = module.configure_torque_reward(cfg)
        self.assertEqual(cfg.joint_torques.weight, -.000625)
        self.assertEqual(cfg.gait.weight, 10.)
        self.assertIsNone(cfg.joint_power)
        self.assertEqual(result["weights_before"], {"joint_torques": -.0005, "gait": 10.})
        self.assertEqual(result["weights_after"], {"joint_torques": -.000625, "gait": 10.})

    def test_unreviewed_original_reward_is_rejected(self):
        module = overlay()
        with self.assertRaises(ValueError):
            module.configure_torque_reward(rewards(-.001))
        cfg = rewards()
        cfg.joint_torques.func.__name__ = "different_function"
        with self.assertRaises(ValueError):
            module.configure_torque_reward(cfg)

    def test_source_keeps_original_plant_and_failure_guards(self):
        module = overlay()
        source = module.build_source(ROOT / "config/train_v15_locomotion.py")
        ast.parse(source)
        self.assertIn("env_cfg.actions.joint_pos.cutoff_hz = 25.0", source)
        self.assertIn("env_cfg.actions.joint_pos.effort_guard_fraction = 0.99", source)
        self.assertIn("make_finite_rollout(make_v13_depth_wrapper())", source)
        self.assertIn('raw_env.unwrapped.reward_manager.get_term_cfg("joint_torques")', source)
        self.assertIn('"reward_changes": True', source)

    def test_lr_reset_preserves_restored_optimizer_state(self):
        module = overlay()
        state = {0: {"exp_avg": [2.]}}
        algorithm = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": .001}], state=state),
                                    learning_rate=.001, schedule="adaptive")
        module.synchronize_fixed_lr(algorithm)
        self.assertIs(algorithm.optimizer.state, state)
        self.assertEqual(algorithm.learning_rate, 3.375e-5)
        self.assertEqual(algorithm.optimizer.param_groups[0]["lr"], 3.375e-5)
        self.assertEqual(algorithm.schedule, "fixed")


if __name__ == "__main__":
    unittest.main()
