"""Join the training logs of the chained segments into one log with each iteration once.

usage: python3 -I merge_training_logs.py OUT_LOG
A segment that hit its time limit logged a few iterations past its last checkpoint; the next segment resumed from
that checkpoint (train_jobs/<next>/checkpoints.sha256, first line), so those extra iterations are dropped here.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEGMENTS = ['10854', '10855', '10856', '10857', '11021', '11022']
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def main():
    merged = []
    for job, following in zip(SEGMENTS, SEGMENTS[1:] + [None]):
        stop = None
        if following:
            resumed = (HERE / 'train_jobs' / following / 'checkpoints.sha256').read_text().split()[1]
            assert f'/v6_robust_train_jobs/{job}/results/model_' in resumed, (following, resumed)
            stop = int(Path(resumed).stem[6:])
        keep = False
        for line in (HERE / 'train_jobs' / job / 'training.log').read_text(errors='replace').splitlines():
            match = re.search(r'Learning iteration (\d+)/', ANSI.sub('', line))
            if match:
                keep = stop is None or int(match.group(1)) <= stop
            if keep:
                merged.append(line)
    Path(sys.argv[1]).write_text('\n'.join(merged) + '\n')


if __name__ == '__main__':
    main()
