"""Pure collection helpers used by the IsaacLab V12 dataset recorder."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from v12_dataset import FIELD_SHAPES, validate_sample_batch


def policy_observation_tensor(observations) -> torch.Tensor:
    """Extract the `[batch,235]` policy group from RSL-RL TensorDict output."""

    if isinstance(observations, torch.Tensor):
        policy = observations
    else:
        try:
            policy = observations["policy"]
        except (KeyError, TypeError) as error:
            raise ValueError("observations are missing the policy group") from error
    if not isinstance(policy, torch.Tensor) or policy.ndim != 2 or policy.shape[1] != 235:
        shape = None if not isinstance(policy, torch.Tensor) else tuple(policy.shape)
        raise ValueError(
            f"policy observation must have shape [batch, 235], got {shape}"
        )
    return policy


def select_rollout_action(
    rollout_policy: str,
    teacher_action: torch.Tensor,
    student_action: torch.Tensor | None,
) -> torch.Tensor:
    """Select the action that drives the simulator without changing labels."""

    if rollout_policy == "teacher":
        return teacher_action
    if rollout_policy == "student":
        if student_action is None:
            raise ValueError("student action is required for student rollout")
        return student_action
    raise ValueError(f"unsupported rollout policy: {rollout_policy}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetShardWriter:
    """Buffer synchronized samples and atomically publish verified shards."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        split: str,
        seed: int,
        target_samples: int,
        shard_size: int,
        teacher_checkpoint_sha256: str,
        camera: dict[str, object],
        metadata: dict[str, object] | None = None,
    ) -> None:
        if target_samples <= 0 or shard_size <= 0:
            raise ValueError("target_samples and shard_size must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.split = split
        self.seed = int(seed)
        self.target_samples = int(target_samples)
        self.shard_size = int(shard_size)
        self.teacher_checkpoint_sha256 = teacher_checkpoint_sha256
        self.camera = dict(camera)
        self.metadata = dict(metadata or {})
        self._parts = {field: [] for field in FIELD_SHAPES}
        self._buffered = 0
        self._accepted = 0
        self._shards: list[dict[str, object]] = []
        self._stats = {
            field: {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}
            for field in FIELD_SHAPES
        }

    @property
    def sample_count(self) -> int:
        return self._accepted

    def append(self, batch: dict[str, torch.Tensor]) -> int:
        """Append at most the remaining target count and return accepted samples."""

        batch_count = validate_sample_batch(batch)
        remaining = self.target_samples - self._accepted
        if remaining <= 0:
            raise ValueError("target sample count is already complete")
        accepted = min(batch_count, remaining)
        for field in FIELD_SHAPES:
            tensor = batch[field][:accepted].detach().to(device="cpu", dtype=torch.float16)
            self._parts[field].append(tensor)
            stats = self._stats[field]
            stats["min"] = min(stats["min"], float(tensor.min()))
            stats["max"] = max(stats["max"], float(tensor.max()))
            stats["sum"] += float(tensor.float().sum())
            stats["count"] += tensor.numel()
        self._buffered += accepted
        self._accepted += accepted
        while self._buffered >= self.shard_size:
            self._flush(self.shard_size)
        return accepted

    def _flush(self, count: int) -> None:
        shard = {}
        for field in FIELD_SHAPES:
            combined = torch.cat(self._parts[field], dim=0)
            shard[field] = combined[:count].contiguous()
            remainder = combined[count:]
            self._parts[field] = [] if remainder.numel() == 0 else [remainder]
        self._buffered -= count

        index = len(self._shards)
        name = f"shard_{index:05d}.pt"
        final_path = self.output_dir / name
        temporary_path = self.output_dir / f".{name}.tmp"
        torch.save(shard, temporary_path)
        os.replace(temporary_path, final_path)
        self._shards.append(
            {
                "file": name,
                "samples": count,
                "sha256": _sha256(final_path),
            }
        )

    def finalize(self) -> dict[str, object]:
        """Flush the exact target count and atomically write its manifest."""

        if self._accepted != self.target_samples:
            raise ValueError(
                f"collected {self._accepted} samples, expected {self.target_samples}"
            )
        if self._buffered:
            self._flush(self._buffered)
        statistics = {
            field: {
                "min": values["min"],
                "max": values["max"],
                "mean": values["sum"] / values["count"],
            }
            for field, values in self._stats.items()
        }
        manifest = {
            "format_version": 1,
            "split": self.split,
            "seed": self.seed,
            "sample_count": self._accepted,
            "target_samples": self.target_samples,
            "shard_size": self.shard_size,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "camera": self.camera,
            "schema": {
                field: {"shape": list(shape), "dtype": "float16"}
                for field, shape in FIELD_SHAPES.items()
            },
            "statistics": statistics,
            "shards": self._shards,
            **self.metadata,
        }
        temporary = self.output_dir / ".manifest.json.tmp"
        final = self.output_dir / "manifest.json"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, final)
        return manifest
