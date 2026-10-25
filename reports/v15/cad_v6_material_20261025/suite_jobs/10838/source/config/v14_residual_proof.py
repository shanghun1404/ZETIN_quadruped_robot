"""Compare first-episode zero-residual evaluation against a frozen-base rollout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare_rollouts(baseline, candidate):
    checks = {key:np.array_equal(baseline[key],candidate[key])
              for key in ('valid','alive_at_end','time_s')}
    errors = {}
    mask = baseline['valid']
    for key in ('positions','quaternions','ground_height','commands','linear_velocity_w','angular_velocity_b'):
        a, b = baseline[key], candidate[key]
        if a.shape != b.shape or not mask.any():
            checks[key] = False
            errors[key] = None
            continue
        error = float(np.abs(a[mask]-b[mask]).max())
        errors[key] = error
        checks[key] = bool(np.isfinite(error) and error <= 1e-6)
    return dict(passed=all(checks.values()), checks=checks, maximum_errors=errors, absolute_tolerance=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    base, candidate = [json.loads(p.read_text()) for p in (args.baseline,args.candidate)]
    bm, cm = base['metadata'], candidate['metadata']
    keys = ('actor_sha256','student_sha256','usd_sha256','seed','scenario','protocol','heading_control',
            'command_forward_m_s','effort_guard_fraction','self_collisions_enabled','leg_terrain_filtered',
            'warmup_s','measured_s','control_dt_s','physics_dt_s','geometry')
    settings_equal = all(bm[k]==cm[k] for k in keys)
    trace_base=np.load(args.baseline.with_suffix('.npz'))
    trace_candidate=np.load(args.candidate.with_suffix('.npz'))
    proof = compare_rollouts(trace_base,trace_candidate)
    residual = cm['residual']
    correction_zero = residual['max_observed_delta_rad']==0 and residual['frozen_base_student_unchanged']
    outcomes_equal = all(base[k]==candidate[k] for k in ('survived_envs','initial_envs','actuation','interleg_clearance','path'))
    proof.update(settings_equal=settings_equal, correction_zero=correction_zero, outcomes_equal=outcomes_equal,
                 base_sha256=bm['actor_sha256'],student_sha256=bm['student_sha256'],usd_sha256=bm['usd_sha256'],
                 baseline=str(args.baseline.resolve()), candidate=str(args.candidate.resolve()),
                 candidate_checkpoint_sha256=residual['sha256'],
                 report_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.baseline,args.candidate)})
    proof['passed'] &= settings_equal and correction_zero and outcomes_equal
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))
    if not proof['passed']:
        raise SystemExit('zero-residual rollout equivalence failed')


if __name__ == '__main__':
    main()
