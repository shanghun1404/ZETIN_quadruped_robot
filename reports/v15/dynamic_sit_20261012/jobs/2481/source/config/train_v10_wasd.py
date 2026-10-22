"""Fine-tune the immutable blind V10.4 checkpoint for a flat WASD demo."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any, Callable, Sequence


TASK_ID = "Isaac-Velocity-Flat-Guide-RobotV2-v10-wasd-demo"
EXPECTED_V10_4_SHA256 = "962518a1b7fe1f62c3f90604cb8f9b49cc2f959c5482e4597a91e9d322df1712"
PERSISTENT_OUTPUT_ROOT = Path(
    "/home/kang/quadruped_training_smooth_v11/runs/v10_flat_wasd_demo"
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _build_parser(*, include_headless: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=_positive_int, default=4096)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_iterations", type=_positive_int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    if include_headless:
        parser.add_argument("--headless", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser(include_headless=True).parse_args(argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_hash(path: str | Path) -> str:
    checkpoint = Path(path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = sha256_file(checkpoint)
    if actual != EXPECTED_V10_4_SHA256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected {EXPECTED_V10_4_SHA256}, got {actual}"
        )
    return actual


def validate_new_output_directory(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    root = PERSISTENT_OUTPUT_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"output must remain beneath {root}") from error
    if candidate == root:
        raise ValueError("output must be a new child directory beneath the persistent root")
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(candidate)
    return candidate


def select_final_checkpoint(
    output_root: str | Path,
    *,
    expected_iteration: int,
) -> Path:
    numbered: list[tuple[int, Path]] = []
    for path in Path(output_root).glob("model_*.pt"):
        match = re.fullmatch(r"model_([0-9]+)\.pt", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise RuntimeError("final checkpoint is missing")
    iteration, checkpoint = max(numbered, key=lambda item: item[0])
    if iteration != expected_iteration:
        raise RuntimeError(
            f"final checkpoint iteration mismatch: expected {expected_iteration}, got {iteration}"
        )
    return checkpoint


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(key) and _finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all().item())
    except ModuleNotFoundError:
        pass
    return True


def checkpoint_is_finite(checkpoint: dict[str, Any]) -> bool:
    state = checkpoint.get("model_state_dict")
    return isinstance(state, dict) and bool(state) and _finite(checkpoint)


def publish_training_candidate(
    final_checkpoint: str | Path,
    output_dir: str | Path,
    report: dict[str, Any],
) -> Path:
    if report.get("finite") is not True or report.get("shapes_valid") is not True:
        raise RuntimeError("candidate is not finite and shape-compatible")
    source = Path(final_checkpoint)
    target = Path(output_dir) / "demo_candidate.pt"
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return target


def lifecycle_status(run: Callable[[], int], cleanup: Callable[[], Any]) -> int:
    """Preserve an active failure/status when simulator cleanup exits zero."""
    status = 0
    active_failure = False
    try:
        status = int(run())
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        active_failure = True
        status = 1
    try:
        cleanup()
    except SystemExit as error:
        close_status = 0 if error.code is None else int(error.code)
        if not active_failure and status == 0 and close_status != 0:
            status = close_status
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        if not active_failure and status == 0:
            status = 1
    return status


def _official_isaac_version() -> str:
    from isaacsim.core.version import get_version

    raw = get_version()
    if len(raw) < 5 or (int(raw[2]), int(raw[3])) != (5, 1):
        raise RuntimeError(f"official Isaac Sim 5.1 is required; found {raw!r}")
    return f"{int(raw[2])}.{int(raw[3])}.{int(raw[4])}"


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner

    from training_resume import load_training_checkpoint
    from v10_wasd_env_cfg import GuideV2V10WASDEnvCfg
    from v10_wasd_ppo_cfg import V10_WASD_PPO_CFG

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V10 WASD training")
    input_sha = verify_checkpoint_hash(args.checkpoint)
    output_root = validate_new_output_directory(args.output_root)
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID is required")
    isaac_version = _official_isaac_version()

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": GuideV2V10WASDEnvCfg,
            "rsl_rl_cfg_entry_point": V10_WASD_PPO_CFG,
        },
    )
    env_cfg = GuideV2V10WASDEnvCfg()
    agent_cfg = copy.deepcopy(V10_WASD_PPO_CFG)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    output_root.mkdir(parents=True, exist_ok=False)
    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=str(output_root),
            device=agent_cfg.device,
        )
        load_training_checkpoint(
            runner,
            args.checkpoint,
            map_location=agent_cfg.device,
        )
        resume_iteration = int(runner.current_learning_iteration)
        if resume_iteration < 0:
            raise RuntimeError("checkpoint did not restore a valid PPO iteration")
        dump_yaml(str(output_root / "params/env.yaml"), env_cfg)
        dump_yaml(str(output_root / "params/agent.yaml"), agent_cfg)
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
        final_iteration = int(runner.current_learning_iteration)
        final_checkpoint = select_final_checkpoint(
            output_root,
            expected_iteration=final_iteration,
        )
        base_env = raw_env.unwrapped
        observation_width = int(
            base_env.observation_manager.compute()["policy"].shape[-1]
        )
        action_width = int(base_env.action_manager.action.shape[-1])
        shapes_valid = observation_width == 48 and action_width == 12
        loaded = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
        finite = isinstance(loaded, dict) and checkpoint_is_finite(loaded)
        report: dict[str, Any] = {
            "finite": finite,
            "shapes_valid": shapes_valid,
            "task_id": TASK_ID,
            "job_id": job_id,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "isaac_version": isaac_version,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "resume_iteration": resume_iteration,
            "final_iteration": final_iteration,
            "checkpoint_input": str(Path(args.checkpoint).resolve()),
            "checkpoint_input_sha256": input_sha,
            "final_checkpoint": str(final_checkpoint.resolve()),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "policy_observation_width": observation_width,
            "action_width": action_width,
        }
        candidate = publish_training_candidate(final_checkpoint, output_root, report)
        report["demo_candidate"] = str(candidate.resolve())
        report["demo_candidate_sha256"] = sha256_file(candidate)
        if not _finite(report):
            raise RuntimeError("training report contains non-finite values")
        (output_root / "training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = _build_parser(include_headless=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    verify_checkpoint_hash(args.checkpoint)
    validate_new_output_directory(args.output_root)
    simulation_app = AppLauncher(args, fast_shutdown=False).app

    def execute() -> int:
        run_training(args)
        return 0

    return lifecycle_status(execute, simulation_app.close)


def process_entry(argv: Sequence[str] | None = None) -> None:
    """Flush evidence and bypass unstable Isaac/Python finalizers."""
    status = 1
    try:
        status = int(main(argv))
    except SystemExit as error:
        status = error.code if isinstance(error.code, int) else 1
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        status = 1
    if status < 0 or status > 255:
        status = 1
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    os._exit(status)


if __name__ == "__main__":
    process_entry()
