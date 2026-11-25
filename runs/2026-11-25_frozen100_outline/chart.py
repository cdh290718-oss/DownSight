"""Paired changes in joint error, including the full range and a close view near zero."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ev=Path(__file__).resolve().parent/'evidence'
rows=json.loads((ev/'results.json').read_text())['rows']
colors={'S1':'#3775a1','S9':'#b7582c','S11':'#338368'}
positions={}
for pair,row in rows.items():
    before,after=[row['methods'][m] for m in ('v3','outline')]
    positions[pair]=np.array([after[k]-before[k] for k in ('pa_mm','nonpa_mm')])
fig,axes=plt.subplots(1,2,figsize=(8.7,4.5))
for ax,title in zip(axes,['All 100 pairs','Detail near zero']):
    for subject,color in colors.items():
        data=np.stack([positions[p] for p in rows if rows[p]['subject']==subject])
        ax.scatter(*data.T,s=20,color=color,alpha=.72,label=subject,edgecolors='none')
    ax.axhline(0,color='#555555',linewidth=.8)
    ax.axvline(0,color='#555555',linewidth=.8)
    ax.set(title=title,xlabel='Change in PA-MPJPE (mm)')
    for name in ('top','right'):ax.spines[name].set_visible(False)
axes[0].set_ylabel('Change in camera-space MPJPE (mm)')
axes[0].legend(frameon=False,loc='lower right')
axes[1].set(xlim=(-4,3),ylim=(-6,4))
for ax,pair,offset in [(axes[0],max(rows,key=lambda k:np.linalg.norm(positions[k])),(3,6)),
                       (axes[1],'pair055',(-40,10)),(axes[1],'pair068',(5,-12))]:
    ax.annotate(pair,positions[pair],xytext=offset,textcoords='offset points',fontsize=8)
fig.suptitle('Stereo contour refinement minus Golden D v3')
fig.text(.09,.02,'Lower left: both joint errors decreased. Each point is one stereo pair.',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95))
fig.savefig(ev/'paired_changes.png',dpi=130)
