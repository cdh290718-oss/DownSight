"""Check gating masks, calibrations, heldout split, and fixed head anchor."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();mode=['baseline','published_gate','recomputed_gate','wrist_only','published_gate_wrist_only','finger_2d_only','finger_3d_only','published_gate_cross_hand'];summary=json.loads((a.inputs/'summary.json').read_text());base=np.load(a.inputs/'baseline/observations.npz',allow_pickle=False);cal=(a.inputs/'baseline/camera_calibration.json').read_bytes();out={}
for m in mode:
 d=np.load(a.inputs/m/'observations.npz',allow_pickle=False);assert (a.inputs/m/'camera_calibration.json').read_bytes()==cal;assert np.array_equal(d['frames'],base['frames']);names=d['names'].tolist();omitted=int(np.sum(d['valid'][10:,1,:]&~d['valid'][10:,0,:]&~d['tri_valid'][10:]));result={'test_cam3_only_not_consumed':omitted,'test_stereo_count':int(d['tri_valid'][10:].sum()),'test_cam2_count':int(d['valid'][10:,0].sum()),'fit_checks':{}}
 if 'wrist_only' in m:
  idx=[i for i,n in enumerate(names) if 'wrist' not in n and 'elbow' not in n];assert not d['valid'][:,:,idx].any() and not d['tri_valid'][:,idx].any()
 if m in ('published_gate','recomputed_gate','published_gate_wrist_only'):
  for row in summary[m]['decisions']:
   i=int(np.where(d['frames']==row['frame'])[0][0]);j=names.index(row['joint'])
   if 'wrist_only' not in m or 'wrist' in row['joint'] or 'elbow' in row['joint']:
    assert bool(d['valid'][i,0,j])==row['used_cam2'] and bool(d['tri_valid'][i,j])==row['used_stereo']
 if m=='finger_2d_only':
  idx=[i for i,n in enumerate(names) if 'wrist' not in n and 'elbow' not in n];assert not d['tri_valid'][:,idx].any() and np.array_equal(d['valid'][:,:,idx],base['valid'][:,:,idx])
 if m=='finger_3d_only':
  idx=[i for i,n in enumerate(names) if 'wrist' not in n and 'elbow' not in n];assert not d['valid'][:,:,idx].any() and np.array_equal(d['tri_valid'][:,idx],base['tri_valid'][:,idx])
 if m=='published_gate_cross_hand':
  published=np.load(a.inputs/'published_gate/observations.npz',allow_pickle=False);assert np.all(d['valid']<=published['valid']) and np.all(d['tri_valid']<=published['tri_valid'])
 for step in (500,1500):
  f=np.load(a.results/f'{m}_{step}'/'test.npz',allow_pickle=False);settings=json.loads((a.results/f'{m}_{step}'/'settings.json').read_text());assert np.array_equal(f['frame_ids'],base['frames'][10:]) and settings['steps']==step;assert np.isfinite(f['joints_cam2']).all();err=np.linalg.norm(f['head_joint_cam2']-f['shared_head_center_cam2_m'],axis=1).max();assert err<1e-6;result['fit_checks'][str(step)]={'head_max_error_m':float(err),'passed':True}
 out[m]=result
with a.output.open('x') as f:json.dump({'status':'passed','checks':out},f,indent=2,allow_nan=False);f.write('\n')
