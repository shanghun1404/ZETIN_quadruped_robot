import hashlib
import json

import pytest
import torch

from v12_dataset import load_dataset_shards, validate_sample_batch


def sample_batch(count: int, offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "depth": torch.full((count, 1, 84, 84), 0.25 + offset, dtype=torch.float16),
        "proprioception": torch.full((count, 48), 1.0 + offset, dtype=torch.float16),
        "teacher_scan": torch.full((count, 187), -0.2 + offset, dtype=torch.float16),
        "teacher_action": torch.full((count, 12), 3.0 + offset, dtype=torch.float16),
    }


def write_dataset(directory, batches, *, sample_count=None):
    directory.mkdir()
    shards = []
    for index, batch in enumerate(batches):
        name = f"shard_{index:05d}.pt"
        path = directory / name
        torch.save(batch, path)
        shards.append(
            {
                "file": name,
                "samples": int(batch["depth"].shape[0]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "sample_count": sum(item["samples"] for item in shards)
        if sample_count is None
        else sample_count,
        "shards": list(reversed(shards)),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_validate_sample_batch_accepts_the_literal_training_schema():
    assert validate_sample_batch(sample_batch(3)) == 3


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("depth", torch.zeros((2, 84, 84)), "depth must have shape"),
        ("proprioception", torch.zeros((2, 47)), "proprioception must have shape"),
        ("teacher_scan", torch.zeros((2, 186)), "teacher_scan must have shape"),
        ("teacher_action", torch.zeros((2, 11)), "teacher_action must have shape"),
    ],
)
def test_validate_sample_batch_rejects_wrong_tensor_shapes(field, replacement, message):
    batch = sample_batch(2)
    batch[field] = replacement

    with pytest.raises(ValueError, match=message):
        validate_sample_batch(batch)


def test_validate_sample_batch_rejects_nonfinite_and_out_of_range_depth():
    nonfinite = sample_batch(2)
    nonfinite["teacher_scan"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="teacher_scan contains non-finite"):
        validate_sample_batch(nonfinite)

    out_of_range = sample_batch(2)
    out_of_range["depth"][0, 0, 0, 0] = 1.01
    with pytest.raises(ValueError, match="depth must be within"):
        validate_sample_batch(out_of_range)


def test_load_dataset_shards_verifies_hashes_and_uses_filename_order(tmp_path):
    directory = tmp_path / "dataset"
    write_dataset(directory, [sample_batch(2, 0.0), sample_batch(1, 0.5)])

    loaded = load_dataset_shards(directory)

    assert loaded["depth"].shape == (3, 1, 84, 84)
    assert loaded["proprioception"].shape == (3, 48)
    assert loaded["teacher_scan"].shape == (3, 187)
    assert loaded["teacher_action"].shape == (3, 12)
    assert torch.equal(loaded["depth"][:2], sample_batch(2, 0.0)["depth"])
    assert torch.equal(loaded["depth"][2:], sample_batch(1, 0.5)["depth"])


def test_load_dataset_shards_rejects_manifest_count_mismatch(tmp_path):
    directory = tmp_path / "dataset"
    write_dataset(directory, [sample_batch(2)], sample_count=3)

    with pytest.raises(ValueError, match="manifest sample_count 3 does not match loaded 2"):
        load_dataset_shards(directory)


def test_load_dataset_shards_rejects_a_tampered_shard(tmp_path):
    directory = tmp_path / "dataset"
    write_dataset(directory, [sample_batch(2)])
    with (directory / "shard_00000.pt").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset_shards(directory)
