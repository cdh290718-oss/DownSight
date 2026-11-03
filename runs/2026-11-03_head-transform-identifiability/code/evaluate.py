"""Read held-out DFM only after all fixed-design unlabelled fits have finished."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser()
p.add_argument('--results',type=Path,required=True); p.add_argument('--reference',type=Path,required=True)
p.add_argument('--observations',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
ref=np.load(a.reference,allow_pickle=False); obs=np.load(a.observations,allow_pickle=False)
lookup={int(f):i for i,f in enumerate(ref['frames'])}
results={}
for condition in ('no_prior','weak_prior','shoulders','stereo_2','stereo_5','no_prior_long'):
 seeds=range(7) if condition in ('no_prior','weak_prior','shoulders') else ([0,1,3,4] if condition=='no_prior_long' else [None])
 rows=[]; centers=[]; rotations=[]; translations=[]
 for seed in seeds:
  name=condition if seed is None else f'{condition}_{seed}'
  directory=a.results/name
  if not (directory/'settings.json').exists(): raise RuntimeError('missing completed fit: '+name)
  settings=json.loads((directory/'settings.json').read_text()); data=np.load(directory/'test.npz',allow_pickle=False)
  ids=[lookup[int(f)] for f in data['frame_ids']]; gt=ref['cam2_body_xyz'][ids]; valid=ref['cam2_body_valid'][ids]
  center=data['shared_head_center_cam2_m']; h=data['head_joint_cam2']; rot=data['shared_rotation_head_to_cam2']; trans=data['shared_camera_in_head_m']
  he=np.linalg.norm(h-gt[:,15],axis=-1)*100; ce=np.linalg.norm(center[None,:]-gt[:,15],axis=-1)*100
  wrists={}
  for side,si,gi in [('left',12,20),('right',13,21)]:
   truth=gt[:,gi]; unit=truth/np.maximum(np.linalg.norm(truth,axis=-1,keepdims=True),1e-8)
   err=np.abs(((data['joints_cam2'][:,si]-truth)*unit).sum(-1))*100
   mask=valid[:,gi]&np.isfinite(err)
   tri_index=obs['names'].tolist().index(side+'_wrist'); mask=mask&obs['tri_valid'][10:,tri_index]
   trierr=np.abs(((obs['tri'][10:,tri_index]-truth)*unit).sum(-1))*100
   wrists[side]={'count':int(mask.sum()),'model_cm':float(np.median(err[mask])) if mask.any() else None,'tri_cm':float(np.median(trierr[mask])) if mask.any() else None,'per_frame_model_cm':err[mask].tolist(),'per_frame_tri_cm':trierr[mask].tolist()}
  rows.append(dict(seed=seed,initial_center_cam2_m=settings['initial_center_cam2_m'],center_cam2_m=center.tolist(),camera_in_head_m=trans.tolist(),rotation_head_to_cam2=rot.tolist(),head_error_cm_median=float(np.median(he[valid[:,15]])),shared_center_error_cm_median=float(np.median(ce[valid[:,15]])),head_error_cm_per_frame=he[valid[:,15]].tolist(),heldout_head_valid_count=int(valid[:,15].sum()),wrists=wrists,train_loss=float(np.load(directory/'train.npz',allow_pickle=False)['final_loss']),test_loss=float(data['final_loss']),shoulder_valid_counts_train_test=settings.get('shoulder_valid_counts_train_test')))
  centers.append(center); rotations.append(rot); translations.append(trans)
 def pairdist(xs):
  return max([float(np.linalg.norm(xs[i]-xs[j])*100) for i in range(len(xs)) for j in range(i)] or [0])
 def pairrot(xs):
  return max([float(np.degrees((Rotation.from_matrix(xs[i]).inv()*Rotation.from_matrix(xs[j])).magnitude())) for i in range(len(xs)) for j in range(i)] or [0])
 results[condition]={'runs':rows,'max_pairwise_center_cm':pairdist(centers),'max_pairwise_camera_in_head_cm':pairdist(translations),'max_pairwise_rotation_deg':pairrot(rotations),'center_coordinate_range_cm':(np.ptp(np.array(centers),axis=0)*100).tolist()}
output={'evaluation_policy':'DFM read only after all prespecified fits completed; no DFM-based fitting or selection','train_frames':obs['frames'][:10].tolist(),'test_frames':obs['frames'][10:].tolist(),'results':results}
with a.output.open('x') as f: json.dump(output,f,indent=2,allow_nan=False); f.write('\n')
