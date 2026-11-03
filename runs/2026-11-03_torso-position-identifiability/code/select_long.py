"""Select most separated fits using only unlabelled training output."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out={}
for c in ('baseline','dense'):
 centers=[]
 for i in range(7):
  with np.load(a.results/f'{c}_{i}'/'train.npz',allow_pickle=False) as d:centers.append(d['shared_head_center_cam2_m'])
 distance,i,j=max((float(np.linalg.norm(centers[i]-centers[j])),i,j) for i in range(7) for j in range(i));out[c]={'seeds':[i,j],'separation_cm':distance*100}
with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
