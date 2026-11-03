"""Post-hoc DFM-only evaluation. Must run after all label-free fitting is complete."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
D=Path('/vepfs-mlp2/mlp-public/huyaoqing/dfm_eval_kit/dfm_reference.npz')
cal=json.loads((ROOT/'camera_calibration.json').read_text());R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);rv=[np.array(cal['cam3_R_virtual_from_gen']),np.array(cal['cam4_R_virtual_from_gen'])]
obs=np.load(ROOT/'observations.npz',allow_pickle=False);dfm=np.load(D,allow_pickle=False);reference_index={int(f):i for i,f in enumerate(dfm['frames'])}
variants=['full','no_shared','no_visibility','no_stereo']
joints=[('left_elbow',10,18),('right_elbow',11,19),('left_wrist',12,20),('right_wrist',13,21)]

def stats(x):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
 if not len(x):return {'count':0,'mean_cm':None,'median_cm':None,'p90_cm':None}
 return dict(count=len(x),mean_cm=float(np.mean(x)*100),median_cm=float(np.median(x)*100),p90_cm=float(np.percentile(x,90)*100))
def metric(pred,truth,valid):
 delta=pred-truth;u=truth/np.maximum(np.linalg.norm(truth,axis=-1,keepdims=True),1e-8);radial=np.sum(delta*u,axis=-1);lateral=np.linalg.norm(delta-radial[...,None]*u,axis=-1);e=np.linalg.norm(delta,axis=-1)
 mask=valid & np.isfinite(delta).all(axis=-1)&np.isfinite(truth).all(axis=-1)
 return dict(along_view_signed=stats(radial[mask]),along_view_absolute=stats(np.abs(radial[mask])),transverse=stats(lateral[mask]),euclidean=stats(e[mask]))
def proxy(head,neck,left,right):
 # Anatomical proxy, not observed head rotation: x shoulders, y neck->head.
 x=right-left; x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8);y=head-neck;y=y-(x*y).sum(-1,keepdims=True)*x;y=y/np.maximum(np.linalg.norm(y,axis=-1,keepdims=True),1e-8);z=np.cross(x,y);z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-8);y=np.cross(z,x)
 return np.stack([x,y,z],axis=-1)
def angle(A,B):
 # Project float32 rotation matrices onto SO(3) before measuring small angles.
 A=np.asarray(A,dtype=np.float64).reshape(-1,3,3).copy();B=np.asarray(B,dtype=np.float64).reshape(-1,3,3).copy()
 return np.degrees((Rotation.from_matrix(A).inv()*Rotation.from_matrix(B)).magnitude())
metrics={'evaluation_policy':'DFM accessed only by this post-fit evaluator; no DFM-guided tuning','splits':{'train':obs['frames'][:10].tolist(),'test':obs['frames'][10:].tolist()},'variants':{},'single_time_diagnostic_baselines':{},'triangulation':{}}
for variant in variants:
 data=np.load(ROOT/('fit_'+variant+'_v2')/'test.npz',allow_pickle=False);settings=json.loads((ROOT/('fit_'+variant+'_v2')/'settings.json').read_text());ids=[reference_index[int(f)] for f in data['frame_ids']];out={'by_camera_and_joint':{},'head_camera':{},'heldout_frame_count':len(ids),'settings':settings}
 j2=data['joints_cam2'];j3=(j2-t)@R
 for c,j in [(2,j2),(3,j3)]:
  gt=dfm['cam%d_body_xyz'%c][ids];ok=dfm['cam%d_body_valid'%c][ids]
  out['by_camera_and_joint']['cam%d'%c]={name:metric(j[:,smpl_idx],gt[:,gt_idx],ok[:,gt_idx]) for name,smpl_idx,gt_idx in joints}
 h=data['head_joint_cam2'];rots=data['head_R_cam2'];hp=dfm['cam2_body_xyz'][ids,15];valid=dfm['cam2_body_valid'][ids,15];pos_errors=np.linalg.norm(h-hp,axis=-1)
 modelproxy=proxy(h,j2[:,7],j2[:,8],j2[:,9]);ref=dfm['cam2_body_xyz'][ids];refproxy=proxy(ref[:,15],ref[:,12],ref[:,16],ref[:,17]);proxy_error=angle(modelproxy,refproxy)
 SR=data['shared_rotation_head_to_cam2'];St=data['shared_camera_in_head_m'];per_t=-np.einsum('nij,nj->ni',np.swapaxes(rots,1,2),h);rot_dev=angle(np.broadcast_to(SR,rots.shape),rots);trans_dev=np.linalg.norm(per_t-St,axis=-1)
 out['head_camera']={'shared_head_to_cam2_rotation_matrix':SR.tolist(),'shared_camera_in_head_m':St.tolist(),'shared_head_center_cam2_m':data['shared_head_center_cam2_m'].tolist(),'per_frame':[],'head_joint_position_error_cm':stats(pos_errors[valid]),'anatomical_proxy_orientation_error_deg_median':float(np.median(proxy_error)), 'anatomical_proxy_note':'Shoulder-neck-head proxy, not DFM head orientation; shoulders unverified', 'per_frame_translation_deviation_cm_median':float(np.median(trans_dev)*100),'per_frame_rotation_deviation_deg_median':float(np.median(rot_dev))}
 for k,f in enumerate(data['frame_ids']):
  out['head_camera']['per_frame'].append(dict(frame=int(f),estimated_head_joint_cam2_m=h[k].tolist(),dfm_head_joint_cam2_m=hp[k].tolist(),dfm_head_joint_valid=bool(valid[k]),joint_error_cm=float(pos_errors[k]*100) if valid[k] else None,estimated_camera_in_head_m=per_t[k].tolist(),translation_deviation_cm=float(trans_dev[k]*100),head_rotation_deviation_deg=float(rot_dev[k]),anatomical_proxy_orientation_error_deg=float(proxy_error[k]),visible_head_vertices_cam2_cam3=data['visible_head_count'][k].tolist()))
 out['visible_head_vertex_count_median_cam2_cam3']=np.median(data['visible_head_count'],axis=0).tolist()
 metrics['variants'][variant]=out
# Existing baseline and label-anchored D are single time point only, excluded from held-out table.
base=Path('/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/gen_egosmplx_two_frames_20261030')
for name,path in [('egosmplx_baseline',base/'output_double_sphere_retry_20261030'),('experiment_D_label_anchored',base/'experiment_D/output_run2')]:
 result={}
 for c,cam,rotation in [(2,'cam3',rv[0]),(3,'cam4',rv[1])]:
  path0=path/'frames'/('gen_00000_'+cam)/'fused/mesh.npz'
  with np.load(path0,allow_pickle=False) as mesh:jj=mesh['body_joints_cam']@rotation
  gt=dfm['cam%d_body_xyz'%c][reference_index[0]];valid=dfm['cam%d_body_valid'%c][reference_index[0]]
  result['cam%d'%c]={n:metric(jj[sm][None,:],gt[gi][None,:],np.array([valid[gi]])) for n,sm,gi in joints}
 metrics['single_time_diagnostic_baselines'][name]={'frame':0,'note':'One frame only; not comparable with held-out aggregate. D uses DFM head labels.', 'results':result}
# Evaluate precomputed stereo wrist positions only after optimization, without using labels for selection.
for side in ['left','right']:
 k=obs['names'].tolist().index(side+'_wrist'); gi=20 if side=='left' else 21;idx=[reference_index[int(f)] for f in obs['frames'][10:]]
 metrics['triangulation'][side+'_wrist_cam2_test']=metric(obs['tri'][10:,k],dfm['cam2_body_xyz'][idx,gi],obs['tri_valid'][10:,k]&dfm['cam2_body_valid'][idx,gi])
with (ROOT/'metrics_v2.json').open('x') as f:json.dump(metrics,f,indent=2,allow_nan=False);f.write('\n')
print('saved metrics_v2.json; held-out frames:',len(metrics['splits']['test']))
