"""Plot captured observations only; no access to Jetson or control interfaces."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
request=json.loads((a.run/'stress_start.json').read_text())
result=json.loads((a.run/'audited_result.json').read_text())
start=request['start_perf_ns'];warm=request['warmup_seconds']
def peaks(rows):
    bins={}
    for t,v in rows:
        if warm<=t<request['seconds']:
            second=int(t-warm);bins[second]=max(v,bins.get(second,0))
    return np.array(sorted(bins))/60,np.array([bins[k] for k in sorted(bins)])
planner=json.loads((a.run/'planner_samples.json').read_text())
ai=peaks((r['stamp_ns']/1e9,r['compute_ms']) for r in planner)
with (a.run/'wbc_samples.csv').open() as f:
    wbc=peaks(((int(r['started_ns'])-start)/1e9,float(r['compute_ms'])) for r in csv.DictReader(f))
os_rows=[json.loads(l) for l in (a.run/'os_samples.jsonl').read_text().splitlines()]
utc=datetime.fromisoformat(request['start_utc'])
os_rows=[r for r in os_rows if warm<=(datetime.fromisoformat(r['utc'])-utc).total_seconds()<=request['seconds']]
times=[((datetime.fromisoformat(r['utc'])-utc).total_seconds()-warm)/60 for r in os_rows]
fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for ax,series,title in ((axs[0,0],ai,'CUDA AI planner'),(axs[0,1],wbc,'C++ reference + WBC')):
    ax.plot(*series,linewidth=.7,label='Maximum in each 1-second bin')
    ax.axhline(20,color='tab:red',linestyle='--',linewidth=1,label='20 ms period')
    ax.set(title=title,ylabel='Compute wall time (ms)');ax.legend(fontsize=8)
zones=sorted(set().union(*(r['temperatures_c'].keys() for r in os_rows)))
selected=[z for z in zones if 'cpu' in z.lower() or 'gpu' in z.lower() or 'tj' in z.lower()]
for name in selected or zones:
    axs[1,0].plot(times,[r['temperatures_c'].get(name,float('nan')) for r in os_rows],label=name,linewidth=1)
axs[1,0].set(title='Recorded temperatures',ylabel='Temperature (C)');axs[1,0].legend(fontsize=8)
axs[1,1].plot(times,[r['memory_available_mb'] for r in os_rows],label='System available memory')
axs[1,1].set(title='Memory headroom',ylabel='Available memory (MiB)')
for ax in axs.flat:
    ax.set(xlabel='Measured time (minutes)',xlim=(0,60));ax.grid(alpha=.2)
duration='completed' if result['execution_duration_completed'] else 'incomplete'
fig.suptitle('Corrected AI + WBC hour workload '+duration+'; integration '+('PASSED' if result['integration_passed'] else 'FAILED'),fontsize=14)
fig.savefig(a.output/'timing_and_resources.png',dpi=160)
plt.close(fig)
