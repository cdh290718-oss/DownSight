"""Structural checks on fit outputs; never reads DFM."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
obs=np.load(a.observations,allow_pickle=False);frames=obs['frames']; names=[f'{c}_{i}' for c in ('no_prior','weak_prior','shoulders') for i in range(7)]+['stereo_2','stereo_5']+[f'no_prior_long_{i}' for i in (0,1,3,4)]
summary=[]
for name in names:
 pth=a.results/name; s=json.loads((pth/'settings.json').read_text())
 train=np.load(pth/'train.npz',allow_pickle=False);test=np.load(pth/'test.npz',allow_pickle=False)
 assert np.array_equal(train['frame_ids'],frames[:10]) and np.array_equal(test['frame_ids'],frames[10:]); assert not set(train['frame_ids'])&set(test['frame_ids'])
 assert s['annotation_used'] is False
 for data in (train,test):
  for key in ('joints_cam2','head_joint_cam2','shared_rotation_head_to_cam2','shared_camera_in_head_m','shared_head_center_cam2_m','final_loss'): assert np.isfinite(data[key]).all()
 assert np.allclose(train['shared_rotation_head_to_cam2'],test['shared_rotation_head_to_cam2']) and np.allclose(train['shared_camera_in_head_m'],test['shared_camera_in_head_m'])
 residual=np.linalg.norm(train['shared_head_center_cam2_m']+train['shared_rotation_head_to_cam2']@train['shared_camera_in_head_m'])
 assert residual<1e-6
 assert s['steps']==(1500 if name.startswith('no_prior_long') else 500)
 summary.append({'name':name,'steps':s['steps'],'shared_center_identity_residual_m':float(residual),'train_frames':train['frame_ids'].tolist(),'test_frames':test['frame_ids'].tolist(),'passed':True})
with a.output.open('x') as f:json.dump({'status':'passed','fit_count':len(summary),'checks':summary},f,indent=2);f.write('\n')
