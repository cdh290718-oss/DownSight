"""Append all-frame stability and proxy-only DFM comparisons to final metrics."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
metrics=json.loads((ROOT/'metrics_v2.json').read_text())
dfm=np.load('/vepfs-mlp2/mlp-public/huyaoqing/dfm_eval_kit/dfm_reference.npz',allow_pickle=False);lookup={int(f):i for i,f in enumerate(dfm['frames'])}
def proxy(head,neck,left,right):
 x=right-left;x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8);y=head-neck;y=y-(x*y).sum(-1,keepdims=True)*x;y=y/np.maximum(np.linalg.norm(y,axis=-1,keepdims=True),1e-8);z=np.cross(x,y);z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-8);y=np.cross(z,x);return np.stack((x,y,z),axis=-1)
def angle(A,B):return np.degrees((Rotation.from_matrix(np.asarray(A,dtype=float).copy()).inv()*Rotation.from_matrix(np.asarray(B,dtype=float).copy())).magnitude())
for variant in ['full','no_shared','no_visibility','no_stereo']:
 rows=[]
 for split in ['train','test']:
  d=np.load(ROOT/('fit_'+variant+'_v2')/(split+'.npz'),allow_pickle=False);frames=d['frame_ids'];idx=[lookup[int(f)] for f in frames];gt=dfm['cam2_body_xyz'][idx];valid=dfm['cam2_body_valid'][idx]
  head=d['head_joint_cam2'];j=d['joints_cam2'];rot=d['head_R_cam2'];Rs=d['shared_rotation_head_to_cam2'];ts=d['shared_camera_in_head_m'];rs=np.repeat(Rs[None],len(frames),axis=0)
  cam_in_head=-np.einsum('nij,nj->ni',rot.transpose(0,2,1),head); dev=np.linalg.norm(cam_in_head-ts,axis=1);deg=angle(rs,rot)
  pred_proxy=proxy(head,j[:,7],j[:,8],j[:,9]);gtproxy=proxy(gt[:,15],gt[:,12],gt[:,16],gt[:,17]);gt_proxy_cam=-np.einsum('nij,nj->ni',gtproxy.transpose(0,2,1),gt[:,15]);pred_proxy_cam=-np.einsum('nij,nj->ni',pred_proxy.transpose(0,2,1),head)
  orient=angle(gtproxy,pred_proxy)
  for i,f in enumerate(frames):
   rows.append(dict(frame=int(f),split=split,head_joint_cam2_m=head[i].tolist(),dfm_head_joint_cam2_m=gt[i,15].tolist(),dfm_head_valid=bool(valid[i,15]),head_joint_error_cm=float(np.linalg.norm(head[i]-gt[i,15])*100) if valid[i,15] else None,head_to_cam2_rotation=rot[i].tolist(),camera_in_head_m=cam_in_head[i].tolist(),translation_difference_from_shared_cm=float(dev[i]*100),rotation_difference_from_shared_deg=float(deg[i]),proxy_camera_position_error_cm=float(np.linalg.norm(pred_proxy_cam[i]-gt_proxy_cam[i])*100),proxy_orientation_error_deg=float(orient[i]),visible_head_vertices_cam2_cam3=d['visible_head_count'][i].tolist()))
 metrics['variants'][variant]['all_frames_stability']={'note':'SMPL-X head coordinate frame; shared transform estimated from train and fixed on test. no_shared has no learned transform; its differences use the inactive initial prior only.', 'proxy_note':'DFM has head/neck/shoulder joints only, no head orientation. Proxy-based comparisons are not head-pose ground truth and shoulders are unverified.', 'rows':rows}
with (ROOT/'metrics_v3.json').open('x') as f:json.dump(metrics,f,indent=2,allow_nan=False);f.write('\n')
print('metrics_v3.json includes all 20 per-frame head-camera stability and post-hoc proxy comparisons')
