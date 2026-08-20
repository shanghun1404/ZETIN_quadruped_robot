import hashlib

import pytest
import torch

from v12_collection import (
    DatasetShardWriter,
    policy_observation_tensor,
    select_rollout_action,
)
from v12_dataset import load_dataset_shards


def sample_batch(count: int, value: float) -> dict[str, torch.Tensor]:
    return {
        "depth": torch.full((count, 1, 84, 84), value, dtype=torch.float16),
        "proprioception": torch.full((count, 48), value, dtype=torch.float16),
        "teacher_scan": torch.full((count, 187), value, dtype=torch.float16),
        "teacher_action": torch.full((count, 12), value, dtype=torch.float16),
    }


def test_shard_writer_splits_batches_atomically_and_writes_a_loadable_manifest(tmp_path):
    output = tmp_path / "train"
    writer = DatasetShardWriter(
        output,
        split="train",
        seed=42,
        target_samples=5,
        shard_size=4,
        teacher_checkpoint_sha256="a" * 64,
        camera={"height": 84, "width": 84},
    )

    writer.append(sample_batch(3, 0.1))
    writer.append(sample_batch(2, 0.2))
    manifest = writer.finalize()

    assert manifest["sample_count"] == 5
    assert manifest["target_samples"] == 5
    assert manifest["split"] == "train"
    assert manifest["seed"] == 42
    assert [entry["samples"] for entry in manifest["shards"]] == [4, 1]
    assert not list(output.glob("*.tmp"))
    for entry in manifest["shards"]:
        shard = output / entry["file"]
        assert entry["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert manifest["statistics"]["depth"]["min"] == pytest.approx(0.1, abs=1e-3)
    assert manifest["statistics"]["depth"]["max"] == pytest.approx(0.2, abs=1e-3)

    loaded = load_dataset_shards(output)
    assert loaded["depth"].shape == (5, 1, 84, 84)
    assert torch.allclose(loaded["depth"][:3].float(), torch.full((3, 1, 84, 84), 0.1), atol=1e-3)
    assert torch.allclose(loaded["depth"][3:].float(), torch.full((2, 1, 84, 84), 0.2), atol=1e-3)


def test_shard_writer_refuses_to_finalize_before_the_target_count(tmp_path):
    writer = DatasetShardWriter(
        tmp_path / "validation",
        split="validation",
        seed=142,
        target_samples=3,
        shard_size=2,
        teacher_checkpoint_sha256="b" * 64,
        camera={"height": 84, "width": 84},
    )
    writer.append(sample_batch(2, 0.1))

    with pytest.raises(ValueError, match="collected 2 samples, expected 3"):
        writer.finalize()


def test_rollout_action_selection_never_substitutes_the_wrong_policy():
    teacher = torch.full((2, 12), 1.0)
    student = torch.full((2, 12), 2.0)

    assert select_rollout_action("teacher", teacher, student) is teacher
    assert select_rollout_action("student", teacher, student) is student
    with pytest.raises(ValueError, match="student action is required"):
        select_rollout_action("student", teacher, None)
    with pytest.raises(ValueError, match="unsupported rollout policy"):
        select_rollout_action("random", teacher, student)


def test_policy_observation_tensor_extracts_the_rsl_policy_group():
    policy = torch.randn((3, 235))

    assert policy_observation_tensor({"policy": policy}) is policy
    assert policy_observation_tensor(policy) is policy
    with pytest.raises(ValueError, match="policy observation must have shape"):
        policy_observation_tensor({"policy": torch.zeros((3, 234))})
