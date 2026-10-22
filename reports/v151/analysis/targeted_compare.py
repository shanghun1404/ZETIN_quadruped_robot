"""Flat-first candidate selection using the original frozen comparison gates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

LABELS=('candidate25','candidate50','candidate100')


def load_api(frozen_source):
    path=Path(frozen_source)/'reports/v151/analysis/compare_ros2.py'
    spec=importlib.util.spec_from_file_location('frozen_v151_comparison',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.frozen_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    return module


def flat_gate(root,label,api):
    baseline=api.load_run(root/'baseline339_flat_seed42')
    candidate=api.load_run(root/f'{label}_flat_seed42')
    gate=api.compare_pair(baseline,candidate)
    report=dict(label=label,baseline=baseline,candidate=candidate,gate=gate,
                frozen_gate_source_sha256=api.frozen_sha256)
    (root/f'{label}_flat_gate.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def summarize(root,api):
    candidates={};eligible=[]
    baseline_rough=api.load_run(root/'baseline339_rough_seed42')
    for label in LABELS:
        flat=flat_gate(root,label,api)
        if not flat['gate']['all_gates_pass']:
            candidates[label]=dict(status='rejected_flat',flat=flat,rough='skipped_after_flat_rejection')
            continue
        rough=api.load_run(root/f'{label}_rough_seed42')
        gate=api.compare_pair(baseline_rough,rough)
        candidates[label]=dict(status='passed_screen' if gate['all_gates_pass'] else 'rejected_rough',
                               flat=flat,rough=dict(candidate=rough,gate=gate))
        if gate['all_gates_pass']:
            score=sum(flat['gate'][k]+gate[k] for k in ('mean_joint_rms_ratio','absolute_power_ratio'))
            eligible.append((score,label))
    selected=min(eligible)[1] if eligible else None
    report=dict(mode='targeted_screen',seed=42,selected=selected,all_gates_pass=bool(selected),
                candidates=candidates,baseline_rough=baseline_rough,
                frozen_gate_source_sha256=api.frozen_sha256,
                adoption_pending_heldout=bool(selected),stairs_validated=False,
                evaluation_source='frozen original screening snapshot592',
                skipped_rough_means='flat gate already failed; not evidence of rough success')
    (root/'comparison_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (root/'selected.txt').write_text((selected or '')+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--frozen-source',type=Path,required=True)
    p.add_argument('--flat',choices=LABELS);args=p.parse_args();api=load_api(args.frozen_source)
    if args.flat:
        result=flat_gate(args.root,args.flat,api)
        print(json.dumps(result['gate']),flush=True)
        raise SystemExit(0 if result['gate']['all_gates_pass'] else 10)
    result=summarize(args.root,api)
    print(json.dumps(dict(selected=result['selected'],all_gates_pass=result['all_gates_pass'])),flush=True)
