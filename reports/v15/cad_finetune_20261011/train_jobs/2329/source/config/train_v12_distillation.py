"""Train TinyDepthViT against sharded V11 scan and frozen-action targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from v12_dataset import FIELD_SHAPES, load_dataset_shards
from v12_teacher_actor import load_frozen_v11_actor
from v12_training import (
    BestMetricTracker,
    EarlyStopper,
    checkpoint_is_finite,
    compute_distillation_batch,
)
from v12_vision_student import TinyDepthViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--extra-train-dir", action="append", default=[])
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_datasets(datasets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        field: torch.cat([dataset[field] for dataset in datasets], dim=0)
        for field in FIELD_SHAPES
    }


def device_batch(
    dataset: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        field: dataset[field][indices].to(device=device, dtype=torch.float32)
        for field in FIELD_SHAPES
    }


class MetricTotals:
    def __init__(self) -> None:
        self.samples = 0
        self.total_loss = 0.0
        self.scan_loss = 0.0
        self.action_loss = 0.0
        self.scan_abs = 0.0
        self.scan_square = 0.0
        self.scan_elements = 0
        self.action_abs = 0.0
        self.action_square = 0.0
        self.action_elements = 0

    def add(self, result, batch) -> None:
        count = int(batch["depth"].shape[0])
        self.samples += count
        self.total_loss += float(result["total"].detach()) * count
        self.scan_loss += float(result["scan"].detach()) * count
        self.action_loss += float(result["action"].detach()) * count
        scan_error = result["predicted_scan"].detach() - batch["teacher_scan"]
        action_error = result["student_action"].detach() - batch["teacher_action"]
        self.scan_abs += float(scan_error.abs().sum())
        self.scan_square += float(scan_error.square().sum())
        self.scan_elements += scan_error.numel()
        self.action_abs += float(action_error.abs().sum())
        self.action_square += float(action_error.square().sum())
        self.action_elements += action_error.numel()

    def metrics(self) -> dict[str, float]:
        if self.samples == 0:
            raise ValueError("cannot report empty metrics")
        return {
            "total_loss": self.total_loss / self.samples,
            "scan_loss": self.scan_loss / self.samples,
            "action_loss": self.action_loss / self.samples,
            "scan_mae": self.scan_abs / self.scan_elements,
            "scan_rmse": math.sqrt(self.scan_square / self.scan_elements),
            "action_mae": self.action_abs / self.action_elements,
            "action_rmse": math.sqrt(self.action_square / self.action_elements),
        }


def evaluate(model, actor, dataset, batch_size, device) -> dict[str, float]:
    model.eval()
    totals = MetricTotals()
    with torch.inference_mode():
        for start in range(0, dataset["depth"].shape[0], batch_size):
            stop = min(start + batch_size, dataset["depth"].shape[0])
            indices = torch.arange(start, stop)
            batch = device_batch(dataset, indices, device)
            result = compute_distillation_batch(model, actor, batch)
            totals.add(result, batch)
    return totals.metrics()


def save_checkpoint(path, model, optimizer, epoch, metrics, metadata) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "architecture": {
            "input": [1, 84, 84],
            "patch_size": 7,
            "patch_tokens": 144,
            "embed_dim": 128,
            "transformer_blocks": 4,
            "attention_heads": 4,
            "feedforward_dim": 256,
            "output_scan_dim": 187,
        },
        "status": "trained_simulator_distillation",
        **metadata,
    }
    if not checkpoint_is_finite(checkpoint):
        raise RuntimeError("refusing to save a non-finite model checkpoint")
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main() -> dict[str, object]:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "epoch_metrics.jsonl"

    train_sources = [Path(args.train_dir), *map(Path, args.extra_train_dir)]
    train_data = merge_datasets([load_dataset_shards(path) for path in train_sources])
    validation_data = load_dataset_shards(args.validation_dir)
    initial = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    initial_state = initial.get("model_state_dict")
    if not isinstance(initial_state, dict):
        raise ValueError("initial state is missing model_state_dict")

    model = TinyDepthViT().to(args.device)
    model.load_state_dict(initial_state, strict=True)
    actor = load_frozen_v11_actor(args.teacher_checkpoint, device=args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    metadata = {
        "seed": args.seed,
        "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint),
        "initial_state_sha256": sha256_file(args.initial_state),
        "train_manifests": [str((path / "manifest.json").resolve()) for path in train_sources],
        "validation_manifest": str((Path(args.validation_dir) / "manifest.json").resolve()),
        "train_samples": int(train_data["depth"].shape[0]),
        "validation_samples": int(validation_data["depth"].shape[0]),
    }

    initial_metrics = evaluate(
        model,
        actor,
        validation_data,
        args.batch_size,
        args.device,
    )
    print(f"[INITIAL] {json.dumps(initial_metrics, sort_keys=True)}", flush=True)
    tracker = BestMetricTracker(initial_metrics, epoch=0)
    stopper = EarlyStopper(
        args.patience,
        best=initial_metrics["total_loss"],
    )
    save_checkpoint(
        output_dir / "best.pt",
        model,
        optimizer,
        0,
        {"train": None, "validation": initial_metrics},
        metadata,
    )
    completed_epochs = 0
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        totals = MetricTotals()
        permutation = torch.randperm(train_data["depth"].shape[0], generator=generator)
        gradient_norm_sum = 0.0
        batches = 0
        for start in range(0, permutation.numel(), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch = device_batch(train_data, indices, args.device)
            optimizer.zero_grad(set_to_none=True)
            result = compute_distillation_batch(model, actor, batch)
            if not torch.isfinite(result["total"]):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            result["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch}")
            optimizer.step()
            gradient_norm_sum += float(gradient_norm)
            batches += 1
            totals.add(result, batch)

        train_metrics = totals.metrics()
        validation_metrics = evaluate(
            model,
            actor,
            validation_data,
            args.batch_size,
            args.device,
        )
        elapsed = time.monotonic() - started
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "mean_gradient_norm_before_clip": gradient_norm_sum / batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_s": elapsed,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[EPOCH] {json.dumps(record, sort_keys=True)}", flush=True)

        completed_epochs = epoch
        improved = tracker.consider(validation_metrics, epoch=epoch)
        should_stop = stopper.update(validation_metrics["total_loss"])
        checkpoint_metrics = {"train": train_metrics, "validation": validation_metrics}
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            checkpoint_metrics,
            metadata,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
                metadata,
            )
        if should_stop:
            print(f"[EARLY_STOP] epoch={epoch}", flush=True)
            break

    best_metrics = tracker.best_metrics
    best_epoch = tracker.best_epoch
    improved_scan = best_metrics["scan_mae"] < initial_metrics["scan_mae"]
    improved_action = best_metrics["action_mae"] < initial_metrics["action_mae"]
    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    report = {
        "status": "training_complete" if improved_scan and improved_action else "training_failed_improvement_gate",
        **metadata,
        "epochs_completed": completed_epochs,
        "best_epoch": best_epoch,
        "initial_validation": initial_metrics,
        "best_validation": best_metrics,
        "improved_scan_mae": improved_scan,
        "improved_action_mae": improved_action,
        "best_checkpoint_finite": checkpoint_is_finite(best_checkpoint),
        "best_checkpoint_sha256": sha256_file(output_dir / "best.pt"),
        "last_checkpoint_sha256": sha256_file(output_dir / "last.pt"),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "training_complete":
        raise RuntimeError(report["status"])
    return report


if __name__ == "__main__":
    main()
