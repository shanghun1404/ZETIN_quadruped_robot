"""Contract tests for the V9 checkpoint sweep.

Two runs now show the same shape on this robot: V8 was better at 1600 than at
8000, and V8.1's duty balance and stance height declined monotonically from
iteration 2000 to 8000 while phase sharpness improved. Judging V9 on its final
checkpoint alone would repeat that a third time, so the sweep is part of the
contract rather than something done by hand afterwards.
"""

import ast
from pathlib import Path

import pytest


SWEEP = "sweep_v9_checkpoints.sh"


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _shell_default(source, name):
    for line in source.splitlines():
        line = line.strip()
        if line.startswith(f"{name}=") and ":-" in line:
            return line.split(":-", 1)[1].split("}", 1)[0]
    raise AssertionError(f"{name} not found in shell source")


def test_sweep_names_checkpoints_that_are_actually_written():
    """model_1999.pt is never written; asking for it sweeps nothing, silently."""
    save_interval = 100
    assert f"save_interval={save_interval}" in _source("v9_ppo_cfg.py")
    for it in (int(x) for x in _shell_default(_source(SWEEP), "SWEEP_ITERS").split()):
        assert it % save_interval == 0, f"model_{it}.pt is never written"


def test_sweep_samples_early_training_where_the_regression_lives():
    iters = [int(x) for x in _shell_default(_source(SWEEP), "SWEEP_ITERS").split()]
    # V8.1's best checkpoint was 2000; the sweep has to reach that far back.
    assert min(iters) <= 2000
    # And it must resolve *between* 2000 and 4000, which V8.1's sweep never did:
    # its quality peaked at its earliest sample, so the true peak could have been
    # anywhere below it.
    assert len(iters) >= 4
    assert any(2000 < it < 4000 for it in iters)


def test_sweep_reports_checkpoints_it_could_not_score():
    source = _source(SWEEP)
    assert "SWEEP_MISSING" in source
    assert "sweep_missing_checkpoint" in source
    assert "sweep_incomplete" in source
    # The bare guard that swallowed the failure in the first V8.1 sweep is gone.
    assert '[[ -f "$ckpt" ]] || continue' not in source


def test_sweep_scores_both_scenarios_per_checkpoint():
    """A terrain result that ignores stairs is not the guide-dog task."""
    source = _source(SWEEP)
    assert "--scenario stairs_up" in source
    # Filenames are version-prefixed so a V9 and a V9.1 sweep cannot overwrite
    # each other's gate JSONs in the shared run_control directory.
    assert "${VER}_gate_${it}.json" in source
    assert "${VER}_stairs_gate_${it}.json" in source
    assert '--env "$VER"' in source
    # The stairs run needs longer than the flat one to actually reach the steps,
    # matching what queue_v9_pilot.sh uses for the final checkpoint.
    assert "--seconds 12" in source
    assert "--seconds 12" in _source("queue_v9_pilot.sh")


def test_sweep_ranking_uses_both_scenarios_and_duty_balance():
    source = _source(SWEEP)
    for key in (
        "rough_gates_passed",
        "stairs_gates_passed",
        "min_foot_contact",
        "stairs_climb_success",
        "hardest_level_cleared",
    ):
        assert key in source
    assert "reverse=True" in source
    assert "${VER}_pilot_sweep.json" in source


def test_sweep_waits_for_the_pilot_instead_of_racing_it():
    """Running during the queue's own gate and video stages contends for the GPU."""
    source = _source(SWEEP)
    assert "stage=waiting_for_pilot" in source
    assert "${VER}_pilot.status" in source
    for terminal in ("stage=passed", "stage=gate_failed", "stage=checkpoint_missing"):
        assert terminal in source
    # It must give up rather than wait forever if the pilot never finishes.
    assert "V9_SWEEP_MAX_WAIT" in source
    assert "timed_out_waiting_for_pilot" in source


def test_sweep_is_separate_from_the_queue_it_observes():
    """bash reads scripts incrementally; editing a running one corrupts it."""
    assert Path(SWEEP).exists()
    # The sweep must not be spliced into the pilot script it waits on.
    assert "sweep_v9_checkpoints" not in _source("queue_v9_pilot.sh")


def test_sweep_records_the_winner_when_it_is_not_the_final_checkpoint():
    source = _source(SWEEP)
    assert "play_v9_video.py" in source
    assert "BEST_ITER" in source
    # The final checkpoint is filmed by the queue already; -1 marks it here.
    assert '"$BEST_ITER" != "-1"' in source
    # Both views the comparison needs: flat gait and the stairs scenario.
    assert source.count("play_v9_video.py") == 2
    assert "--scenario stairs_up --view side" in source


@pytest.mark.parametrize("marker", ["set -uo pipefail", "write_status"])
def test_sweep_follows_the_queue_script_conventions(marker):
    assert marker in _source(SWEEP)


def test_sweep_defaults_to_the_live_version_not_the_stopped_one():
    """V9 stalled, V9.1 declined, V9.2 stood still; V9.3 is what to score."""
    source = _source(SWEEP)
    assert 'VER="${V9_SWEEP_VERSION:-v9_3}"' in source
    assert 'EXPERIMENT="${V9_SWEEP_EXPERIMENT:-guide_${VER}_terrain}"' in source
    # Evaluation must run under the config the checkpoint trained with.
    assert 'evaluate_v9.py --headless --env "$VER"' in source
    assert 'play_v9_video.py --env "$VER"' in source
