"""CPU tests for evaluation input binding; no checkpoint is executed."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "config/experiments/v16_20260929/v15_torque_reward_eval_manifest.py"


def implementation():
    assert MODULE.exists(), "Evaluation manifest implementation missing"
    spec = importlib.util.spec_from_file_location("eval_manifest", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root):
    actor = root / "original.pt"
    actor.write_bytes(b"CPU test fixture; not a trained policy")
    (root / "v15_torque_reward_finetune.py").write_bytes((ROOT / "config/experiments/v16_20260929/v15_torque_reward_finetune.py").read_bytes())
    original_sha = hashlib.sha256(actor.read_bytes()).hexdigest()
    log = root / "guide_v15_torque_reward_finetune" / "fixture"
    log.mkdir(parents=True)
    checkpoint = log / "model_11949.pt"
    checkpoint.write_bytes(b"second CPU test fixture")
    contract = {"algorithm_optimizer_lr_match": True, "algorithm_learning_rate_after": 3.375e-5,
                "optimizer_group_lr_after": [3.375e-5], "schedule_after": "fixed",
                "actual_guard_fraction": .99, "actual_cutoff_hz": 25., "starting_learning_iteration": 11700,
                "num_learning_iterations": 250, "num_envs": 512, "seed": 47,
                "rollout_steps_per_env": 24, "episode_length_s": 20., "physics_dt_s": .005,
                "control_dt_s": .02, "max_delay_steps": 3, "physics_diverged_termination": True,
                "actual_lpf_alpha": .005 / (.005 + 1 / (2 * math.pi * 25))}
    reward_contract = {"weights_before": {"joint_torques": -.0005, "gait": 10.},
                       "weights_after": {"joint_torques": -.000625, "gait": 10.},
                       "torque_reward_multiplier": 1.25, "joint_torques_weight_before": -.0005,
                       "joint_torques_weight_after": -.000625}
    contract.update(torque_reward_contract=reward_contract, actual_joint_torques_weight=-.000625,
                    actual_joint_torques_function="isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.rewards.joint_torques_penalty",
                    actual_reward_weights=reward_contract["weights_after"])
    contract_path = log / "finetune_actual_contract.json"
    contract_path.write_text(json.dumps(contract))
    completed = {"training_completed": True, "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 "last_learning_iteration": 11949, "nonfinite_steps": 0, "policy_std_min_max": [.1, .2],
                 "actual_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest()}
    (log / "run_completed.json").write_text(json.dumps(completed))
    (log / "v15_manifest.json").write_text(json.dumps({"initial_checkpoint_sha256": original_sha,
                 "resume": True, "reward_changes": True, "torque_reward_contract": reward_contract,
                 "original_trainer_sha256": "3da8cdcefa600f851795d3d48cdd30a6172b55070556ff8d888971057a1391d2", "action_lpf_hz": 25., "effort_guard_fraction": .99,
                 "overlay_sha256": "87db5a6165e5017cb3e1c684ddb905d0ff835b8980c4c2320c72843f5e9f88e4"}))
    template = {"baseline": {"actor_sha256": original_sha}, "candidates": [
        {"label": "adapt_torque1p25_g099_lpf25", "actor_sha256": None}]}
    return actor, template, log, checkpoint


class EvalInputTests(unittest.TestCase):
    def test_trained_actor_is_pinned_separately_from339(self):
        module = implementation()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor, template, log, checkpoint = fixture(root)
            result = module.bind_inputs(root, template, actor)
            self.assertNotEqual(result["manifest"]["baseline"]["actor_sha256"], result["manifest"]["candidates"][0]["actor_sha256"])
            self.assertEqual(result["model_paths"]["adapt_torque1p25_g099_lpf25"], str(checkpoint.resolve()))

    def test_modified_checkpoint_is_rejected(self):
        module = implementation()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor, template, log, checkpoint = fixture(root)
            checkpoint.write_bytes(b"changed after completion")
            with self.assertRaises(RuntimeError):
                module.bind_inputs(root, template, actor)

    def test_changed_other_reward_is_rejected_even_with_updated_marker_hash(self):
        module = implementation()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor, template, log, checkpoint = fixture(root)
            path = log / "finetune_actual_contract.json"
            contract = json.loads(path.read_text())
            contract["torque_reward_contract"]["weights_after"]["gait"] = 5.
            contract["actual_reward_weights"]["gait"] = 5.
            path.write_text(json.dumps(contract))
            marker = log / "run_completed.json"
            completed = json.loads(marker.read_text())
            completed["actual_contract_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            marker.write_text(json.dumps(completed))
            with self.assertRaises(RuntimeError):
                module.bind_inputs(root, template, actor)

    def test_nonfinite_rollout_is_rejected(self):
        module = implementation()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actor, template, log, checkpoint = fixture(root)
            path = log / "run_completed.json"
            completed = json.loads(path.read_text())
            completed["nonfinite_steps"] = 1
            path.write_text(json.dumps(completed))
            with self.assertRaises(RuntimeError):
                module.bind_inputs(root, template, actor)


if __name__ == "__main__":
    unittest.main()
