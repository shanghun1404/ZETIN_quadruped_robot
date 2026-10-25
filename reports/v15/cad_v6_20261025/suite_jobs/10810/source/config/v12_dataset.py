"""Validated sharded tensor storage for V12 depth distillation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


FIELD_SHAPES = {
    "depth": (1, 84, 84),
    "proprioception": (48,),
    "teacher_scan": (187,),
    "teacher_action": (12,),
}


def validate_sample_batch(batch: dict[str, torch.Tensor]) -> int:
    """Validate one stored sample batch and return its batch size."""

    if not isinstance(batch, dict):
        raise TypeError("sample batch must be a dictionary")
    missing = [field for field in FIELD_SHAPES if field not in batch]
    if missing:
        raise ValueError(f"sample batch is missing fields: {missing}")

    sample_count = None
    for field, trailing_shape in FIELD_SHAPES.items():
        tensor = batch[field]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{field} must be a torch.Tensor")
        expected_rank = len(trailing_shape) + 1
        if tensor.ndim != expected_rank or tuple(tensor.shape[1:]) != trailing_shape:
            shape_text = ", ".join(str(value) for value in trailing_shape)
            raise ValueError(f"{field} must have shape [batch, {shape_text}]")
        if sample_count is None:
            sample_count = int(tensor.shape[0])
        elif tensor.shape[0] != sample_count:
            raise ValueError("all sample fields must have the same batch length")
        if not tensor.is_floating_point():
            raise TypeError(f"{field} must be floating point")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{field} contains non-finite values")

    if sample_count is None or sample_count <= 0:
        raise ValueError("sample batch must not be empty")
    depth = batch["depth"]
    if float(depth.min()) < 0.0 or float(depth.max()) > 1.0:
        raise ValueError("depth must be within [0, 1]")
    scan = batch["teacher_scan"]
    if float(scan.min()) < -1.0 or float(scan.max()) > 1.0:
        raise ValueError("teacher_scan must be within [-1, 1]")
    return sample_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_shards(directory: str | Path) -> dict[str, torch.Tensor]:
    """Load, verify, validate, and concatenate a manifest-backed dataset."""

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_entries = manifest.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ValueError("manifest shards must be a non-empty list")

    collected = {field: [] for field in FIELD_SHAPES}
    loaded_count = 0
    for entry in sorted(shard_entries, key=lambda item: item["file"]):
        shard_path = directory / entry["file"]
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        expected_hash = entry.get("sha256")
        if expected_hash is not None and _sha256(shard_path) != expected_hash:
            raise ValueError(f"checksum mismatch for {entry['file']}")
        batch = torch.load(shard_path, map_location="cpu", weights_only=True)
        batch_count = validate_sample_batch(batch)
        if "samples" in entry and int(entry["samples"]) != batch_count:
            raise ValueError(
                f"manifest shard count for {entry['file']} does not match {batch_count}"
            )
        loaded_count += batch_count
        for field in FIELD_SHAPES:
            collected[field].append(batch[field])

    manifest_count = int(manifest.get("sample_count", -1))
    if manifest_count != loaded_count:
        raise ValueError(
            f"manifest sample_count {manifest_count} does not match loaded {loaded_count}"
        )
    return {
        field: torch.cat(parts, dim=0)
        for field, parts in collected.items()
    }
