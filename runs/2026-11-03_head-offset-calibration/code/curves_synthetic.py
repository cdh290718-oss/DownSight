"""Render synthetic per-axis movement sweeps; numbers only, no Gen imagery."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();s=json.loads(a.summary.read_text())
with (a.output_dir/'synthetic_curves.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['axes','per_axis_amplitude_deg','completed_trials','spread_cm_trial0','spread_cm_trial1','worst_true_error_cm_trial0','worst_true_error_cm_trial1','max_actual_angle_deg'])
 for c in s['conditions']:
  vals=lambda key:[*c[key],None,None][:2]
  w.writerow([c['axes'],c['per_axis_amplitude_deg'],c['completed'],*vals('spread_cm'),*vals('worst_start_error_cm'),max(c['max_actual_neck_rotation_deg']) if c['max_actual_neck_rotation_deg'] else ''])
plt.rcParams.update({'svg.fonttype':'none','font.size':9});fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for axes,label in ((1,'One axis'),(2,'Two axes')):
 cases=[c for c in s['conditions'] if c['axes']==axes];x=[c['per_axis_amplitude_deg'] for c in cases];
 for i,key in enumerate(('spread_cm','worst_start_error_cm')):
  y=[max(c[key]) if c[key] else float('nan') for c in cases];ax[i].plot(x,y,marker='o',label=label)
for a1 in ax:a1.axhline(2,color='black',linestyle='--',linewidth=1,label='2 cm reference');a1.set_xlabel('Per-axis amplitude (degrees)');a1.set_xticks([0,10,20,40,60]);a1.grid(alpha=.25);a1.legend(fontsize=8)
ax[0].set(title='Worst multistart spread',ylabel='Maximum pairwise distance (cm)');ax[1].set(title='Worst distance to synthetic truth',ylabel='Maximum error over starts (cm)');fig.suptitle('SMPL-X simulation: worst of two trials, ideal correspondences');fig.savefig(a.output_dir/'synthetic_motion.svg');plt.close(fig)
