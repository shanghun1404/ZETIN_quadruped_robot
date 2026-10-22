"""Export knee RMS figures from the immutable comparison output."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--candidate',required=True)
args=p.parse_args()
root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
entry=next(c for c in summary['candidates'] if c['name']==args.candidate)
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,key,title in zip(axes.flat,
    ['INITIAL_STAND','WALK','POSTSTOP_STAND','FINAL_CROUCH'],
    ['Initial stand','Walking','Stand after stopping','Final crouch (26.5–28 s)']):
    x=np.arange(4)
    for offset,run,label,color in [(-.18,summary['baseline'],'Baseline 601','#7b8794'),
                                  (.18,entry['run'],f"Candidate {entry['run']['run']}",'#1879b9')]:
        values=[j['rms_nm'] for j in run['windows'][key]['joints'] if j['joint'].endswith('_KFE')]
        bars=ax.bar(x+offset,values,width=.36,label=label,color=color)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    ax.axhline(18,color='#b13b37',ls='--',lw=1,label='18 Nm continuous rating')
    ax.set(title=title,xticks=x,xticklabels=['FL','FR','RL','RR'],ylabel='Knee torque RMS [Nm]',ylim=(0,34))
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
axes[0,0].legend(loc='upper left',fontsize=8)
verdict='ALL FROZEN GATES PASSED' if entry['comparison']['accepted'] else 'REJECTED: '+', '.join(entry['comparison']['failed_gates'])
fig.suptitle(f'V15.1 ROS2 / Isaac Sim — flat, seed 42, 28 s\n{verdict}',fontsize=13)
fig.supxlabel('Feedforward command / instantaneous PD torque estimate; not measured motor torque',fontsize=9)
for extension in ('png','svg'):
    fig.savefig(root/f"knees_{entry['run']['run']}.{extension}",dpi=170)
