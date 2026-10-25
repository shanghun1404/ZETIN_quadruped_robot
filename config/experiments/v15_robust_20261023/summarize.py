"""Summarize the robust fine-tune: training curve, robustness eval and the nine-case suite.

Usage: python summarize.py OUT_DIR TRAINING_LOG ROBUST_EVAL_JSON [ROBUST_EVAL_JSON ...]
       --suite LABEL=SUITE_JSON[,SUITE_JSON...] [--suite ...]
Runs with the standard library plus matplotlib; no IsaacLab or torch.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ANSI = re.compile(r'\x1b\[[0-9;]*m')
FIELDS = {
    'Mean reward': 'reward', 'Mean episode length': 'episode_length',
    'Curriculum/robust_ramp': 'robust_level',
    'Episode_Termination/base_tilt': 'base_tilt', 'Episode_Termination/low_height': 'low_height',
    'Episode_Termination/time_out': 'time_out', 'Episode_Termination/tile_exit': 'tile_exit',
    'Metrics/base_velocity/error_vel_xy': 'error_vel_xy', 'Mean policy_retention loss': 'retention_loss',
    'Iteration time': 'iteration_s',
}


def training_rows(path):
    rows, row = [], None
    for line in Path(path).read_text(errors='replace').splitlines():
        line = ANSI.sub('', line).strip()
        match = re.match(r'Learning iteration (\d+)/(\d+)', line)
        if match:
            row = {'iteration': int(match.group(1))}
            rows.append(row)
            continue
        if row is None or ':' not in line:
            continue
        key, _, value = line.rpartition(':')
        key = key.strip()
        if key in FIELDS:
            try:
                row[FIELDS[key]] = float(value.strip().rstrip('s'))
            except ValueError:
                pass
    return rows


def suite_cases(paths):
    cases = {}
    for path in paths:
        for case in json.loads(Path(path).read_text())['cases']:
            cases[case['case']] = dict(task_success=bool(case['task_success']),
                rated_exceed=sum(bool(flag) for flag in case['rms_exceeds_rated']),
                seconds=case['seconds_recorded'], max_tilt_deg=case.get('max_abs_tilt_deg'),
                failure=case.get('failure'), reasons=case.get('motion_verdict', {}).get('reasons', []))
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out');p.add_argument('log');p.add_argument('evals', nargs='+')
    p.add_argument('--suite', action='append', default=[])
    args = p.parse_args()
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    rows = training_rows(args.log)
    evals = {}
    for path in args.evals:
        record = json.loads(Path(path).read_text())
        for label, scenarios in record['results'].items():
            for name, result in scenarios.items():
                evals.setdefault(label, {})[name] = dict(
                    falls=result['falls'], falls_per_robot_minute=result['falls_per_robot_minute'],
                    fall_fraction_of_ended_episodes=result['fall_fraction_of_ended_episodes'],
                    planar_error_mps=result['mean_planar_velocity_error_mps'],
                    yaw_error_radps=result['mean_yaw_rate_error_radps'],
                    tile_exit=result['terminations'].get('tile_exit', 0), robot_seconds=result['robot_seconds'],
                    nonfinite_steps=result['nonfinite_steps'], settings=result['settings'])
    suites = {}
    for item in args.suite:
        label, _, paths = item.partition('=')
        suites[label] = suite_cases(paths.split(','))
    summary = dict(schema=1, training_iterations=len(rows),
        training_first=rows[0] if rows else None, training_last=rows[-1] if rows else None,
        robust_eval=evals, suite=suites)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')

    if rows:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        x = [row['iteration'] for row in rows]
        for axis, key, title in ((axes[0, 0], 'reward', 'Mean reward'), (axes[0, 1], 'episode_length', 'Mean episode length'),
                                 (axes[1, 0], 'robust_level', 'Robust ramp level'), (axes[1, 1], 'base_tilt', 'Base-tilt termination share')):
            axis.plot(x, [row.get(key, float('nan')) for row in rows], lw=1)
            axis.set_title(title);axis.set_xlabel('iteration');axis.grid(alpha=.3)
        fig.savefig(out/'training_curve.png', dpi=130);plt.close(fig)
    if evals:
        labels = list(evals)
        scenarios = list(next(iter(evals.values())))
        fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        width = .8/len(labels)
        for i, label in enumerate(labels):
            axis.bar([j+i*width for j in range(len(scenarios))],
                     [evals[label][name]['falls_per_robot_minute'] for name in scenarios], width, label=label)
        axis.set_xticks([j+width*(len(labels)-1)/2 for j in range(len(scenarios))], scenarios, rotation=30, ha='right')
        axis.set_ylabel('falls per robot-minute');axis.legend();axis.grid(axis='y', alpha=.3)
        fig.savefig(out/'robust_falls.png', dpi=130);plt.close(fig)

    lines = ['| 시나리오 | ' + ' | '.join(f'{label} 넘어짐/로봇·분' for label in evals) + ' | '
             + ' | '.join(f'{label} 속도오차 m/s' for label in evals) + ' |']
    lines.append('|' + '---|'*(1+2*len(evals)))
    for name in (next(iter(evals.values())) if evals else {}):
        lines.append(f'| {name} | ' + ' | '.join(f"{evals[l][name]['falls_per_robot_minute']:.3f} ({evals[l][name]['falls']})" for l in evals)
                     + ' | ' + ' | '.join(f"{evals[l][name]['planar_error_mps']:.3f}" for l in evals) + ' |')
    if suites:
        names = sorted({name for cases in suites.values() for name in cases})
        lines += ['', '| 시험 | ' + ' | '.join(suites) + ' |', '|' + '---|'*(1+len(suites))]
        for name in names:
            cells = []
            for label in suites:
                case = suites[label].get(name)
                cells.append('—' if case is None else ('통과' if case['task_success'] else '미통과')
                             + f" (정격 RMS 초과 {case['rated_exceed']})")
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
