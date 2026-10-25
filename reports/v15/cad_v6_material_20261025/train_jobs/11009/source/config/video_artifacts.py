import math
import shutil
import tempfile
from pathlib import Path


# 0.15 = pilot protocol, 0.40 = release protocol, 0.50 = the V5.11 reference
# walking speed (the guide_rough reference video regime).
SUPPORTED_SPEEDS = ((0.15, "speed015"), (0.40, "speed040"), (0.50, "speed050"))
SPEED_TOLERANCE = 1e-9


def semantic_speed_tag(semantic_speed: float) -> str:
    for supported_speed, tag in SUPPORTED_SPEEDS:
        if math.isclose(
            semantic_speed,
            supported_speed,
            rel_tol=0.0,
            abs_tol=SPEED_TOLERANCE,
        ):
            return tag
    supported = ", ".join(f"{speed:.2f}" for speed, _ in SUPPORTED_SPEEDS)
    raise ValueError(f"speed must be exactly one of: {supported}")


def create_fresh_recording_dir(output_dir: str | Path, prefix: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=output_path))


def copy_exact_recording(
    record_dir: str | Path,
    prefix: str,
    final_path: str | Path,
) -> Path:
    recorded_path = Path(record_dir) / f"{prefix}-step-0.mp4"
    if not recorded_path.is_file():
        raise RuntimeError(f"Recorder produced no exact recording artifact: {recorded_path}")
    destination = Path(final_path)
    shutil.copy2(recorded_path, destination)
    return destination
