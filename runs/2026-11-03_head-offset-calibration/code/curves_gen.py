"""Aggregate all six signed axis perturbations; generate numbers and vector-only plots."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();data=json.loads(a.metrics.read_text());r=data['results'];out=[]
for mag in (0,1,2,5,10):
 names=['zero'] if mag==0 else [f'{axis}_{sign}_{mag}' for axis in 'xyz' for sign in 'pm'];entry={'offset_error_cm':mag,'case_names':names,'metrics':{}}
 for joint in ('pelvis','neck','left_elbow','right_elbow','left_wrist','right_wrist'):
  per={}
  for metric in ('euclidean_vs_dfm','radial_vs_dfm','displacement_from_zero_fit'):
   vals={n:r[n]['joints'][joint][metric]['median_cm'] for n in names};per[metric]={'min_cm':min(vals.values()),'max_cm':max(vals.values()),'worst_case':max(vals,key=vals.get)}
  entry['metrics'][joint]=per
 out.append(entry)
with (a.output_dir/'gen_curves.json').open('x') as f:json.dump({'note':'At each magnitude, maxima/minima are over six axis-aligned signed perturbations, each statistic is median over held-out frames; zero is one case.','curve':out},f,indent=2);f.write('\n')
with (a.output_dir/'gen_curves.csv').open('x',newline='') as f:
 w=csv.writer(f);w.writerow(['offset_error_cm','joint','metric','minimum_median_cm','maximum_median_cm','worst_case'])
 for row in out:
  for joint,metrics in row['metrics'].items():
   for metric,value in metrics.items():w.writerow([row['offset_error_cm'],joint,metric,value['min_cm'],value['max_cm'],value['worst_case']])
plt.rcParams.update({'svg.fonttype':'none','font.size':9})
fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);xx=[x['offset_error_cm'] for x in out]
for name,label in [('pelvis','Pelvis'),('neck','Neck'),('left_elbow','L elbow'),('right_elbow','R elbow'),('left_wrist','L wrist'),('right_wrist','R wrist')]:
 ax[0].plot(xx,[r['metrics'][name]['displacement_from_zero_fit']['max_cm'] for r in out],marker='o',label=label)
for name,label in [('left_wrist','L wrist'),('right_wrist','R wrist')]:ax[1].plot(xx,[r['metrics'][name]['euclidean_vs_dfm']['max_cm'] for r in out],marker='o',label=label)
ax[0].set(title='Change relative to zero offset',ylabel='Max directional median displacement (cm)');ax[1].axhline(2,color='black',linestyle='--',linewidth=1,label='2 cm');ax[1].set(title='Wrist 3D difference vs DFM',ylabel='Max directional median difference (cm)')
for x in ax:x.set_xlabel('Imposed offset magnitude (cm)');x.set_xticks(xx);x.grid(alpha=.25);x.legend(fontsize=8)
fig.suptitle('Held-out Gen frames: signed axis perturbations, not a guarantee');fig.savefig(a.output_dir/'gen_sensitivity.svg');plt.close(fig)
