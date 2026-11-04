"""Validate exact frame coverage and preservation of v0 wrists/elbows without DFM."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--parts',type=Path,required=True);p.add_argument('--v0',type=Path,required=True);p.add_argument('--evaluation-fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();dest=a.evaluation_fits;dest.mkdir(parents=True,exist_ok=False)
for name in ('v0','wilor'):(dest/name).symlink_to((a.v0.parent/name).resolve(),target_is_directory=True)
(dest/'two_stage').mkdir();stats=[];seen=[];per_joint={x:[] for x in ('left_elbow','right_elbow','left_wrist','right_wrist')};noobs=np.zeros(2,dtype=int);noobs_joint_drift={'left':[],'right':[]}
for i in range(0,180,10):
 folder='first90_v2' if i<90 else 'last90_v2';path=a.parts/folder/f'batch_{i:03d}.npz';old=a.v0/f'batch_{i:03d}.npz'
 with np.load(path,allow_pickle=False) as x,np.load(old,allow_pickle=False) as b:
  assert np.array_equal(x['frame_ids'],b['frame_ids']);assert np.isfinite(x['joints_cam2']).all();d=np.linalg.norm(x['joints_cam2'][:,[10,11,12,13]]-b['joints_cam2'][:,[10,11,12,13]],axis=-1);assert d.max()<1e-5
  for j,key in enumerate(per_joint):per_joint[key].extend(d[:,j].tolist())

  for si,side in enumerate(('left','right')):
   mask=x['no_observations'][:,si];assert np.array_equal(x[f'{side}_hand_pose'][mask],b[f'{side}_hand_pose'][mask]);assert np.all(x['wrist_rotation_delta'][mask,si]==0);hand_ids=[12 if si==0 else 13]+list(range(25 if si==0 else 45,45 if si==0 else 65));hand_diff=np.linalg.norm(x['joints_cam2'][:,hand_ids]-b['joints_cam2'][:,hand_ids],axis=-1);noobs_joint_drift[side].extend([{'frame':int(f),'max_point_difference_m':float(q)} for f,q in zip(x['frame_ids'][mask],hand_diff[mask].max(axis=1))])
  noobs+=x['no_observations'].sum(0);seen+=x['frame_ids'].tolist();stats.append({'batch':i,'frames':x['frame_ids'].tolist(),'max_arm_diff_m':float(d.max()),'initial_reconstruction_max_m':float(x['initial_reconstruction_max_m']),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 (dest/'two_stage'/path.name).symlink_to(path.resolve())
assert len(seen)==len(set(seen))==179 and set(seen)==set(range(0,5400,30))-{120}
out={'batch_count':len(stats),'unique_predicted_frames':len(seen),'missing_frame_ids':[120],'max_joint_drift_m':{k:float(np.max(v)) for k,v in per_joint.items()},'mean_joint_drift_m':{k:float(np.mean(v)) for k,v in per_joint.items()},'hand_without_any_2d_or_stereo_observation_counts':{'left':int(noobs[0]),'right':int(noobs[1])},'unobserved_hand_joint_drift':{side:{'max_m':max((x['max_point_difference_m'] for x in rows),default=0.),'per_frame':rows} for side,rows in noobs_joint_drift.items()},'batches':stats,'dfm_used':False};a.output.write_text(json.dumps(out,indent=2)+'\n')
