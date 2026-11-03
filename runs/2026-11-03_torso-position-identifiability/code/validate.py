"""Check split, fixed 3D shared center, finite values, and no shared rotation."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();obs=np.load(a.observations,allow_pickle=False);sel=json.loads((a.results/'long_selection.json').read_text());out=[]
for cond in ('baseline','dense','baseline_long','dense_long'):
 seeds=sel[cond[:-5]]['seeds'] if cond.endswith('_long') else range(7)
 for seed in seeds:
  d=a.results/f'{cond}_{seed}';s=json.loads((d/'settings.json').read_text());tr=np.load(d/'train.npz',allow_pickle=False);te=np.load(d/'test.npz',allow_pickle=False)
  assert np.array_equal(tr['frame_ids'],obs['frames'][:10]) and np.array_equal(te['frame_ids'],obs['frames'][10:]) and not set(tr['frame_ids'])&set(te['frame_ids'])
  assert not s['shared_rotation'] and s['weights']['shared_rotation']==0 and s['weights']['head_prior']==0 and not s['annotation_used']
  assert s['dense']==cond.startswith('dense') and s['steps']==(1500 if cond.endswith('_long') else 500)
  assert np.array_equal(tr['shared_head_center_cam2_m'],te['shared_head_center_cam2_m'])
  for item in (tr,te):
   for k in ('joints_cam2','head_joint_cam2','vertices_cam2','shared_head_center_cam2_m','final_loss'): assert np.isfinite(item[k]).all()
  out.append({'condition':cond,'seed':seed,'train_frames':tr['frame_ids'].tolist(),'test_frames':te['frame_ids'].tolist(),'passed':True})
with a.output.open('x') as f:json.dump({'status':'passed','results':out},f,indent=2);f.write('\n')
