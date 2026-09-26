"""Summarize frozen-base residual evaluations without hiding early failures."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
reports = [(p,json.loads(p.read_text())) for p in sorted(root.glob('*_seed*.json'))]
rows=[]
for p,r in reports:
    m=r['metadata']; residual=m.get('residual')
    assert m['protocol']=='fixed_v2' and m['heading_control']
    assert m['effort_guard_fraction']==.99 and m['self_collisions_enabled'] and m['leg_terrain_filtered']
    if residual:
        assert residual['frozen_base_student_unchanged']
        assert residual['max_observed_delta_rad'] <= .050001
    failures=[e['first_termination_s'] for e in r['per_environment'] if e['first_termination_s'] is not None]
    row=dict(label=p.stem,scenario=m['scenario'],seed=m['seed'],survived=r['survived_envs'],total=r['initial_envs'],
             directed_stairs=r['path']['directed_stair_success_count'],full_flight=r['path']['full_flight_success_count'],
             progress_m=r['path']['mean_final_directed_progress_m'],cross_track_m=r['path']['mean_max_abs_cross_track_m'],
             heading_deg=r['path']['mean_max_abs_heading_error_deg'],
             median_failure_s=float(np.median(failures)) if failures else None,
             minimum_gap_m=r['interleg_clearance']['minimum_gap_m'],
             overlap_fraction=r['interleg_clearance']['negative_gap_frame_fraction'],
             deep_overlap_fraction=r['interleg_clearance']['gap_below_minus_2mm_frame_fraction'],
             valid_gap_frames=r['interleg_clearance']['measured_env_frames'],
             torque_over_limit=r['actuation']['over_limit_request_fraction'],
             max_correction_rad=residual['max_observed_delta_rad'] if residual else 0.,
             residual_sha256=residual['sha256'] if residual else None)
    rows.append(row)
    print(f"{p.stem}: survival {row['survived']}/{row['total']}, directed {row['directed_stairs']}, progress {row['progress_m']:.3f}m, cross {row['cross_track_m']:.3f}m, overlap<-2mm {100*row['deep_overlap_fraction']:.4f}%, max delta {row['max_correction_rad']:.5f}rad")
for key in ('actor_sha256','student_sha256','usd_sha256'):
    assert len({r['metadata'][key] for _,r in reports}) <= 1
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
by_name={row['label']:row for row in rows}
required=[f'{label}_stairs_up_seed{seed}' for label in ('fixed_heading','residual174_early') for seed in (42,43)]
if all(name in by_name for name in required):
    comparison=[]
    for label in ('fixed_heading','residual174_early'):
        selected=[by_name[f'{label}_stairs_up_seed{seed}'] for seed in (42,43)]
        frames=sum(r['valid_gap_frames'] for r in selected)
        combined=dict(configuration=label,seeds=[42,43],total=sum(r['total'] for r in selected),
                      survived=sum(r['survived'] for r in selected),
                      directed_stairs=sum(r['directed_stairs'] for r in selected),
                      full_flight=sum(r['full_flight'] for r in selected),valid_gap_frames=frames,
                      minimum_gap_m=min(r['minimum_gap_m'] for r in selected))
        for key in ('progress_m','cross_track_m','heading_deg'):
            combined[key]=sum(r[key]*r['total'] for r in selected)/combined['total']
        for key in ('overlap_fraction','deep_overlap_fraction'):
            combined[key]=sum(r[key]*r['valid_gap_frames'] for r in selected)/frames
        comparison.append(combined)
    (root/'comparison_2seeds.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print('Two-seed comparison:',json.dumps(comparison))
stairs=[(p,r) for p,r in reports if r['metadata']['scenario']=='stairs_up' and r['metadata']['seed']==42
        and 'zero' not in p.name and p.with_suffix('.npz').exists()]
if stairs:
    fig,axes=plt.subplots(1,len(stairs),figsize=(5*len(stairs),5),squeeze=False,sharex=True,sharey=True)
    for ax,(p,r) in zip(axes[0],stairs):
        t=np.load(p.with_suffix('.npz'))
        for i in range(t['valid'].shape[1]):
            xy=t['positions'][t['valid'][:,i],i,:2]-t['positions'][0,i,:2]
            good=r['path']['per_environment'][i]['directed_stair_success']
            ax.plot(xy[:,0],xy[:,1],color='#238b45' if good else '#c34b48',alpha=.5,lw=1)
        ax.axvspan(-.6,.6,color='#74a9cf',alpha=.13)
        ax.axhline(2.1,color='#333',ls='--',lw=1)
        ax.set_title(p.stem.replace('_stairs_up_seed42','')+f"\n{r['survived_envs']}/32 alive, {r['path']['directed_stair_success_count']}/32 passes")
        ax.set(xlabel='Lateral displacement (m)',xlim=(-4,4),ylim=(-4,7)); ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Forward displacement (m)')
    fig.suptitle('V14 residual: fixed 15.7 cm stairs, first episode only')
    fig.tight_layout(); fig.savefig(root/'paths_seed42.png',dpi=160); plt.close(fig)
