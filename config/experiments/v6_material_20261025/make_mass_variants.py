"""Isaac USD variants of v6 with the leg critical parts (thigh clamps com_pro, hip yoke main + rear incl. BL, knee cup + rear cap)
made of another material: same geometry, so each occurrence's mass and principal inertia scale by rho_new / rho_PLA
(PLA solid 1.24 g/cm3, the density of the model; checked: STL volume x 1.24 = model mass), CoM unchanged.

Run: python3 make_mass_variants.py <isaac_usd_v6 dir> <out dir>
"""
import json, shutil, subprocess, sys
from pathlib import Path

CRIT = ('com_pro', 'hip_yoke_main_v5', 'hip_yoke_rear_v5', 'hip_yoke_main_v5_BL', 'hip_yoke_rear_v5_BL', 'knee', 'knee_rear_cap_v5')
RHO_PLA = 1.24
VARIANTS = {'al7075': 2.81, 'tc4': 4.43, 'ss316l': 7.9}     # g/cm3 (7075-T6; Ti-6Al-4V; SLM 316L)

src, out = Path(sys.argv[1]), Path(sys.argv[2])
for name, rho in VARIANTS.items():
    d = out / f'isaac_usd_v6_{name}'
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d)
    model = json.loads((d / 'model.json').read_text(encoding='utf-8'))
    r = rho / RHO_PLA
    before = after = 0.0
    for link, occ in model['links'].items():
        for o in occ:
            before += o['mass_kg']
            if o['component'] in CRIT:
                o['mass_kg'] *= r
                o['principal_kgcm2'] = [v * r for v in o['principal_kgcm2']]
            after += o['mass_kg']
    (d / 'model.json').write_text(json.dumps(model, indent=1), encoding='utf-8')
    log = subprocess.run([sys.executable, '-I', str(d / 'build_usd.py')], capture_output=True, text=True, cwd=d)
    tail = [l for l in log.stdout.splitlines() if 'total mass' in l or 'link' in l]
    print(f'{name}: rho {rho} (x{r:.3f})  total {before:.3f} -> {after:.3f} kg (+{after - before:.3f})  rc={log.returncode}')
    print('   ' + '\n   '.join(tail[:14]))
    if log.returncode:
        print(log.stderr[-1500:])
