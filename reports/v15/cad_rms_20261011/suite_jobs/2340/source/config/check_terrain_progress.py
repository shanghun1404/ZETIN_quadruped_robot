"""Decide the terrain-curriculum prediction early, from the running training log.

Why this exists
---------------
V9 burned 2 h 43 min proving a curriculum was stuck that had been stuck since
iteration 500. This reads the rsl_rl console log the trainer is already writing,
so the verdict costs nothing and needs no second simulator.

Why it checks a trend and not a threshold
-----------------------------------------
The first version of this checker asked one question at one iteration: "is the
level above 0.5?" V9.1 answered 0.9065 at iteration 1511 and **passed** -- while
the curriculum had already peaked at 1.584 around iteration 500 and was falling
monotonically::

    500 -> 1.584   1000 -> 1.317   1500 -> 0.913   2000 -> 0.716

A single point cannot distinguish "climbing through 0.9" from "falling through
0.9". So the level is now read as a series, and a sustained decline from the peak
fails even when the absolute value clears the bar.

Exit code 0 once a verdict is reached (pass or fail), 1 while the run has not yet
reached the check iteration. The queue polls it and stops polling on 0.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# These deliberately duplicate ``v9_2_env_cfg`` rather than importing it. That
# module imports isaaclab, which requires a running simulator app, and this
# checker is invoked as a plain subprocess from the training queue once a minute
# with no simulator in the process. Importing would crash on every poll.
# ``test_v9_2_command_band_contract`` parses both files and fails if they drift.
TERRAIN_LEVEL_CHECK_ITER = 1500
TERRAIN_LEVEL_MIN_EXPECTED = 1.5
TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK = 0.25
# Iterations before this are startup transient, not curriculum behaviour, and
# are excluded from the peak. rsl_rl runs with ``init_at_random_ep_len=True``,
# so at iteration 0 a large fraction of environments reset immediately and the
# logged level reflects those truncated episodes -- V9.1 and V9.2 both reported
# exactly 0.987 there. Left in, that transient can exceed a healthy plateau and
# fail a run that is doing nothing wrong.
TERRAIN_LEVEL_WARMUP_ITER = 250

# V9's measured band. Anything inside it means nothing has changed at all.
V9_STALLED_BAND = (0.14, 0.21)

ITER_RE = re.compile(r"Learning iteration (\d+)/")
LEVEL_RE = re.compile(r"terrain_levels:\s*([0-9.]+)")


def read_series(log_path: Path):
    """Return [(iteration, terrain_level), ...] in log order.

    rsl_rl prints the iteration header before the metric block it summarises, so
    each level is paired with the most recent iteration line above it.
    """
    series = []
    iteration = None
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            m = ITER_RE.search(line)
            if m:
                iteration = int(m.group(1))
                continue
            m = LEVEL_RE.search(line)
            if m and iteration is not None:
                series.append((iteration, float(m.group(1))))
    return series


def verdict(series, check_iter, min_expected, max_decline, warmup=TERRAIN_LEVEL_WARMUP_ITER):
    """None while it is too early to say; otherwise a decided result dict."""
    if not series:
        return None
    iteration, level = series[-1]
    if iteration < check_iter:
        return None

    # Measure the peak over settled behaviour only; see TERRAIN_LEVEL_WARMUP_ITER.
    settled = [pair for pair in series if pair[0] >= warmup] or series
    peak_iter, peak = max(settled, key=lambda pair: pair[1])
    decline = 0.0 if peak <= 0.0 else (peak - level) / peak

    above_bar = level >= min_expected
    declining = decline > max_decline
    stalled = V9_STALLED_BAND[0] <= level <= V9_STALLED_BAND[1]
    passed = above_bar and not declining

    if passed:
        conclusion = "terrain curriculum is promoting and holding"
    elif stalled:
        conclusion = (
            "terrain curriculum still pinned in V9's band -- the change did not "
            "affect promotion at all"
        )
    elif declining and above_bar:
        conclusion = (
            f"terrain curriculum peaked at {peak:.3f} (iter {peak_iter}) and has fallen "
            f"{decline:.0%} -- promotion is losing to demotion, as in V9.1"
        )
    elif declining:
        conclusion = (
            f"terrain curriculum peaked at {peak:.3f} (iter {peak_iter}) and fell below "
            f"the {min_expected} bar -- the command band still cannot earn promotion"
        )
    else:
        conclusion = (
            f"terrain curriculum is stable at {level:.3f} but below the {min_expected} "
            "bar -- partial effect"
        )

    return {
        "iteration": iteration,
        "terrain_level": level,
        "peak_level": peak,
        "peak_iteration": peak_iter,
        "decline_from_peak": round(decline, 4),
        "check_iteration": check_iter,
        "min_expected": min_expected,
        "max_decline_from_peak": max_decline,
        "warmup_iterations": warmup,
        "above_bar": above_bar,
        "declining": declining,
        "still_in_v9_stalled_band": stalled,
        "passed": passed,
        "conclusion": conclusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Early terrain-curriculum verdict.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", default="terrain_progress.json")
    parser.add_argument("--check-iter", type=int, default=TERRAIN_LEVEL_CHECK_ITER)
    parser.add_argument("--min-level", type=float, default=TERRAIN_LEVEL_MIN_EXPECTED)
    parser.add_argument(
        "--max-decline", type=float, default=TERRAIN_LEVEL_MAX_DECLINE_FROM_PEAK
    )
    parser.add_argument(
        "--warmup", type=int, default=TERRAIN_LEVEL_WARMUP_ITER,
        help="iterations excluded from the peak as startup transient",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        return 1

    series = read_series(log_path)
    result = verdict(series, args.check_iter, args.min_level, args.max_decline, args.warmup)
    if result is None:
        if not args.quiet:
            latest = series[-1] if series else (None, None)
            print(f"[WAIT] iteration={latest[0]} terrain_level={latest[1]}")
        return 1

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"[TERRAIN] {'PASS' if result['passed'] else 'FAIL'} {result['conclusion']}")
    print(
        f"[TERRAIN] iter={result['iteration']} level={result['terrain_level']:.3f} "
        f"peak={result['peak_level']:.3f}@{result['peak_iteration']} "
        f"decline={result['decline_from_peak']:.0%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
