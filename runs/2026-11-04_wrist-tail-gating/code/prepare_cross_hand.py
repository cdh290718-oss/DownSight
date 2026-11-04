"""Exploratory cross-hand fingertip filter; no reference labels or identity reassignment."""
import argparse,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();src=a.inputs/'published_gate';d=np.load(src/'observations.npz',allow_pickle=False);x={k:d[k].copy() for k in d.files};names=d['names'].tolist();rows=[]
for i,frame in enumerate(d['frames']):
 for side in ('left','right'):
  other='right' if side=='left' else 'left';w=names.index(side+'_wrist');wo=names.index(other+'_wrist')
  for finger in ('thumb4','forefinger4','middle_finger4','ring_finger4','pinky_finger4'):
   name=side+'_'+finger;j=names.index(name);usable=bool(d['tri_valid'][i,j] and d['tri_valid'][i,w] and d['tri_valid'][i,wo]);same=float(np.linalg.norm(d['tri'][i,j]-d['tri'][i,w])) if usable else None;opposite=float(np.linalg.norm(d['tri'][i,j]-d['tri'][i,wo])) if usable else None;drop=bool(usable and same>.15 and same-opposite>.05)
   if drop:x['valid'][i,:,j]=False;x['tri_valid'][i,j]=False
   rows.append({'frame':int(frame),'joint':name,'same_wrist_m':same,'opposite_wrist_m':opposite,'dropped':drop,'rule':'same>0.15m and opposite closer by >0.05m'})
dest=a.inputs/'published_gate_cross_hand';dest.mkdir(exist_ok=False);(dest/'camera_calibration.json').write_text((src/'camera_calibration.json').read_text());os.symlink((src/'initial').resolve(),dest/'initial',target_is_directory=True)
with (dest/'observations.npz').open('xb') as f:np.savez_compressed(f,**x)
with a.output.open('x') as f:json.dump({'policy':'post-hoc candidate; suspect observations removed, never relabelled','decisions':rows},f,indent=2,allow_nan=False);f.write('\n')
print('test dropped',[(r['frame'],r['joint']) for r in rows if r['frame'] in set(d['frames'][10:].tolist()) and r['dropped']])
