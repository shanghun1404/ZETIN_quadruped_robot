import json
from pathlib import Path
from statistics import mean

repo = Path(__file__).resolve().parents[2]
rows = []
actors = set()
for scenario in ('flat', 'rough', 'stairs_up', 'stairs_down'):
    old, new = [], []
    source_json = []
    for seed in (42, 43, 44):
        a = json.loads((repo / f'reports/v13/data/v11_actor_guard_0.99_{scenario}_seed{seed}.json').read_text())
        label = 'm5700_verified' if seed == 43 and scenario in ('flat', 'rough') else 'm5700'
        relative_path = f'data/{label}_{scenario}_seed{seed}.json'
        b = json.loads((repo / 'reports/v14' / relative_path).read_text())
        source_json.append(relative_path)
        for key in ('seed', 'warmup_s', 'measured_s', 'control_dt_s', 'physics_dt_s', 'command_forward_m_s', 'effort_guard_fraction', 'scenario', 'terrain_level', 'student_sha256'):
            assert a['metadata'][key] == b['metadata'][key], (scenario, seed, key)
        assert b['metadata']['seed'] == seed
        assert b['metadata']['scenario'] == scenario
        assert b['metadata']['self_collisions_enabled'] is True
        assert b['metadata']['leg_terrain_filtered'] is True
        assert b['metadata']['usd_sha256'] == '864f4b11fc24af1099acf292ce5358192294426928387c6d903cba41d45050ad'
        assert a['initial_envs'] == b['initial_envs'] == 32
        actors.add(b['metadata']['actor_sha256'])
        old.append(a)
        new.append(b)
    def summarize(items):
        result = {
            'total_envs': sum(x['initial_envs'] for x in items),
            'survived': sum(x['survived_envs'] for x in items),
            'traversed': sum(x.get('stairs', {}).get('traversed_count', 0) for x in items) if scenario.startswith('stairs_') else None,
            'seed_mean_progress_m': mean(x['mean_directed_progress_m'] for x in items),
            'seed_mean_joint_power_w': mean(x['actuation']['mean_joint_power_w'] for x in items),
            'max_seed_over_limit_request_fraction': max(x['actuation']['over_limit_request_fraction'] for x in items),
        }
        if all('interleg_clearance' in x for x in items):
            gaps = [x['interleg_clearance'] for x in items]
            frames = sum(g['measured_env_frames'] for g in gaps)
            result['interleg_clearance'] = {
                'measured_env_frames': frames,
                'minimum_gap_m': min(g['minimum_gap_m'] for g in gaps),
                'negative_gap_frame_fraction': sum(g['negative_gap_frame_fraction'] * g['measured_env_frames'] for g in gaps) / frames,
                'gap_below_minus_2mm_frame_fraction': sum(g['gap_below_minus_2mm_frame_fraction'] * g['measured_env_frames'] for g in gaps) / frames,
            }
        return result
    rows.append({'scenario': scenario, 'v13': summarize(old), 'v14_m5700': summarize(new), 'v14_source_json': source_json})
assert actors == {'009cb093951868dfc042b0bac4bc0939d921857311b5988e046667457b2d27bd'}
result = {'candidate': 'pilot110/model_5700.pt', 'actor_sha256': next(iter(actors)), 'seeds': [42, 43, 44], 'note': 'Progress/power are means of the three seed-level metrics. Gap fractions weight alive environment frames. Evaluation protocol metadata matched against V13.', 'terrains': rows}
(repo / 'reports/v14/validation_summary.json').write_text(json.dumps(result, indent=2) + '\n')
for r in rows:
    print(json.dumps(r))
