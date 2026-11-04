import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();obs=np.load(a.inputs/'baseline/observations.npz',allow_pickle=False);rows=[]
for mode in ('baseline','published_gate_cross_hand'):
 for steps in (500,1500):
  name=f'first10_{mode}_{steps}';d=np.load(a.results/name/'test.npz',allow_pickle=False);s=json.loads((a.results/name/'settings.json').read_text());assert np.array_equal(d['frame_ids'],obs['frames'][:10]) and s['steps']==steps and s['anchor']=='hard';e=float(np.linalg.norm(d['head_joint_cam2']-d['shared_head_center_cam2_m'],axis=1).max());assert e<1e-6;rows.append({'name':name,'frame_ids':d['frame_ids'].tolist(),'head_anchor_max_error_m':e,'passed':True})
with a.output.open('x') as f:json.dump({'status':'passed','checks':rows},f,indent=2);f.write('\n')
