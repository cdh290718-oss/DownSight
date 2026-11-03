"""DFM post-fit evaluation and displacement relative to the zero-offset fit."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ref=np.load(a.reference,allow_pickle=False);obs=np.load(a.observations,allow_pickle=False);lookup={int(f):i for i,f in enumerate(ref['frames'])};base=np.load(a.results/'zero/test.npz',allow_pickle=False);base_long=np.load(a.results/'long_zero/test.npz',allow_pickle=False);frames=obs['frames'][10:];assert np.array_equal(base['frame_ids'],frames);idx=[lookup[int(f)] for f in frames];gt=ref['cam2_body_xyz'][idx];valid=ref['cam2_body_valid'][idx];joints=[('pelvis',0,0),('neck',7,12),('left_shoulder',8,16),('right_shoulder',9,17),('left_elbow',10,18),('right_elbow',11,19),('left_wrist',12,20),('right_wrist',13,21)]
def stats(x):
 x=np.asarray(x,dtype=float);return {'count':len(x),'median_cm':float(np.median(x)*100) if len(x) else None,'p90_cm':float(np.percentile(x,90)*100) if len(x) else None,'per_frame_cm':(x*100).tolist()}
headcal=np.load(a.results/'zero/test.npz',allow_pickle=False)['shared_head_center_cam2_m'];hm=valid[:,15]&np.isfinite(gt[:,15]).all(1);calibration_consistency=stats(np.linalg.norm(gt[:,15]-headcal[None,:],axis=1)[hm]);tridx=[lookup[int(f)] for f in obs['frames'][:10]];train_heads=ref['cam2_body_xyz'][tridx,15];train_valid=ref['cam2_body_valid'][tridx,15]&np.isfinite(train_heads).all(1);train_calibration_consistency=stats(np.linalg.norm(train_heads-headcal[None,:],axis=1)[train_valid]);results={}
for folder in sorted(a.results.iterdir()):
 if not folder.is_dir() or not (folder/'settings.json').exists():continue
 d=np.load(folder/'test.npz',allow_pickle=False);comparison=base_long if folder.name.startswith('long_') else base;s=json.loads((folder/'settings.json').read_text());assert np.array_equal(d['frame_ids'],frames);pred=d['joints_cam2'];j={}
 for name,si,gi in joints:
  gtj=gt[:,gi];mask=valid[:,gi]&np.isfinite(gtj).all(1)&np.isfinite(pred[:,si]).all(1);unit=gtj/np.maximum(np.linalg.norm(gtj,axis=1,keepdims=True),1e-8);delta=pred[:,si]-gtj;rad=np.abs((delta*unit).sum(1));eu=np.linalg.norm(delta,axis=1);dis=np.linalg.norm(pred[:,si]-comparison['joints_cam2'][:,si],axis=1)
  j[name]={'valid_dfm_frames':frames[mask].tolist(),'euclidean_vs_dfm':stats(eu[mask]),'radial_vs_dfm':stats(rad[mask]),'displacement_from_zero_fit':stats(dis[np.isfinite(dis)])}
 err=np.max(np.linalg.norm(d['head_joint_cam2']-d['shared_head_center_cam2_m'][None,:],axis=1))
 results[folder.name]={'perturbation_cm':s['perturbation_cm'],'fixed_center_cam2_m':s['shared_head_center_cam2_m'],'hard_anchor_max_error_m':float(err),'loss':float(d['final_loss']),'joints':j}
with a.output.open('x') as f:json.dump({'note':'DFM used once for simulated calibration from training head positions and here as post-fit reference only. Euclidean wrist metric is 3D distance, radial is view-direction absolute error. DFM elbow/torso reference unvalidated.','test_frames':frames.tolist(),'training_dfm_head_to_simulated_calibration':train_calibration_consistency,'heldout_dfm_head_to_simulated_calibration':calibration_consistency,'results':results},f,indent=2,allow_nan=False);f.write('\n')
