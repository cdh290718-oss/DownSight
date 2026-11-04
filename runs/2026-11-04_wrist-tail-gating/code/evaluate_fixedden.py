"""Evaluate fixed-denominator controls against held-out DFM reference."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ref=np.load(a.reference,allow_pickle=False);obs=np.load(a.inputs/'baseline/observations.npz',allow_pickle=False);frames=obs['frames'][10:];indices={int(f):i for i,f in enumerate(ref['frames'])};ii=[indices[int(x)] for x in frames];gt=ref['cam2_body_xyz'][ii];valid=ref['cam2_body_valid'][ii];out={}
for mode in ('wrist_only','finger_2d_only','finger_3d_only','published_gate_cross_hand'):
 norm=mode.startswith('published');normal=np.load(a.inputs/('published_gate' if norm else 'baseline')/'observations.npz',allow_pickle=False);den={'2d':int(normal['valid'][10:,0].sum()),'3d':int(normal['tri_valid'][10:].sum())}
 for step in (500,1500):
  name=f'{mode}_fixedden_{step}';folder=a.results/name;d=np.load(folder/'test.npz',allow_pickle=False);s=json.loads((folder/'settings.json').read_text());assert np.array_equal(d['frame_ids'],frames) and s['fixed_normalization_counts']==den and s['steps']==step
  row={'normalization':den,'sides':{}}
  for side,si,gi in [('left',12,20),('right',13,21)]:
   e=d['joints_cam2'][:,si]-gt[:,gi];mask=valid[:,gi]&np.isfinite(e).all(1);vals=np.linalg.norm(e,axis=1)[mask]*100;row['sides'][side]={'count':len(vals),'median_cm':float(np.median(vals)),'p90_cm':float(np.percentile(vals,90)),'valid_frames':frames[mask].tolist(),'per_frame_cm':vals.tolist()}
  out[name]=row
with a.output.open('x') as f:json.dump({'note':'Denominators fixed to baseline or published-gate count; DFM only post-fit reference.','results':out},f,indent=2,allow_nan=False);f.write('\n')
