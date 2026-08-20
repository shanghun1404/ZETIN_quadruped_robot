import os

import pytest

from video_artifacts import (
    copy_exact_recording,
    create_fresh_recording_dir,
    semantic_speed_tag,
)


def test_copy_requires_exact_current_step_trigger_artifact(tmp_path):
    prefix = "grounded_model_rear_quarter_speed040"
    stale_dir = tmp_path / "old-run"
    stale_dir.mkdir()
    (stale_dir / f"{prefix}-step-0.mp4").write_bytes(b"old exact artifact")

    record_dir = create_fresh_recording_dir(tmp_path, prefix)
    next_record_dir = create_fresh_recording_dir(tmp_path, prefix)
    assert record_dir != stale_dir
    assert next_record_dir != record_dir
    assert not any(record_dir.iterdir())
    assert not any(next_record_dir.iterdir())
    future_candidate = record_dir / f"{prefix}-step-999.mp4"
    future_candidate.write_bytes(b"future-dated collision")
    future_time = 4_102_444_800
    os.utime(future_candidate, (future_time, future_time))
    final_path = tmp_path / f"{prefix}.mp4"

    with pytest.raises(RuntimeError, match="exact recording artifact"):
        copy_exact_recording(record_dir, prefix, final_path)
    assert not final_path.exists()

    exact_artifact = record_dir / f"{prefix}-step-0.mp4"
    exact_artifact.write_bytes(b"current exact artifact")
    copied_path = copy_exact_recording(record_dir, prefix, final_path)

    assert copied_path == final_path
    assert final_path.read_bytes() == b"current exact artifact"


def test_semantic_speed_tag_supports_pilot_release_and_reference_only():
    assert semantic_speed_tag(0.15) == "speed015"
    assert semantic_speed_tag(0.40) == "speed040"
    assert semantic_speed_tag(0.40 + 5e-10) == "speed040"
    # V5.11 records at the reference walking speed.
    assert semantic_speed_tag(0.50) == "speed050"
    with pytest.raises(ValueError, match="must be exactly one of"):
        semantic_speed_tag(0.60)
