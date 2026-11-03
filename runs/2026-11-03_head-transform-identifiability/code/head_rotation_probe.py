"""Perturb the unobserved SMPL-X head joint; no DFM input."""
import os,sys,json,copy,argparse
from pathlib import Path
import numpy as np, torch, cv2
from scipy.spatial.transform import Rotation
repo=Path(os.environ['EGOSMPLX_BACKEND']); source=Path(os.environ['DOUBLE_SPHERE_SOURCE']); old=Path(os.environ['STEREO_EXPERIMENT'])
sys.path.insert(0,str(source)); os.environ['EGOSMPLX_REPO_ROOT']=str(repo);os.environ['EGOSMPLX_CONFIG']=str(repo/'main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py')
from egosmplx_pipeline.bootstrap import setup
smpl,_=setup(repo)
p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
fit=np.load(a.fit/'train.npz',allow_pickle=False); obs=np.load(old/'observations.npz',allow_pickle=False); cal=json.loads((old/'camera_calibration.json').read_text()); n=len(fit['frame_ids'])
layer=copy.deepcopy(smpl.layer['neutral']).cuda().eval();layer.requires_grad_(False)
T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
raw=[]
for frame in fit['frame_ids']:
 with np.load(old/'initial/egosmplx/smplx'/('gen_%05d_00_smplx.npz'%frame),allow_pickle=False) as d:raw.append({k:d[k].copy() for k in d.files})
keys=['betas','jaw_pose','leye_pose','reye_pose','expression']
fixed={k:T(np.stack([r[k].reshape(-1) for r in raw])) for k in keys}
fixed.update(left_hand_pose=T(fit['left_hand_pose']),right_hand_pose=T(fit['right_hand_pose']))
body=fit['body_pose'].copy().reshape(n,21,3); modified=body.copy()
q=Rotation.from_euler('x',5,degrees=True);modified[:,14]= (Rotation.from_rotvec(body[:,14])*q).as_rotvec()
headids=np.where(smpl.layer['neutral'].lbs_weights.detach().cpu().numpy().argmax(1)==smpl.orig_joints_name.index('Head'))[0]
obsidx=[smpl.joints_name.index(x) for x in obs['smpl_names'].tolist()]
def model(b):
 with torch.no_grad():
  out=layer(body_pose=T(b.reshape(n,-1)),global_orient=T(fit['global_orient']),**fixed)
  return (out.joints[:,smpl.joint_idx]+T(fit['transl'])[:,None,:]).cpu().numpy(), (out.vertices+T(fit['transl'])[:,None,:]).cpu().numpy()[:,headids]
def visible(v):
 result=[];r=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32'])
 for points,ds in [(v,cal['cam2_ds']),((v-t)@r,cal['cam3_ds'])]:
  fx,fy,cx,cy,xi,alpha=ds;x,y,z=np.moveaxis(points,-1,0);d1=np.sqrt(x*x+y*y+z*z+1e-12);z1=xi*d1+z;d2=np.sqrt(x*x+y*y+z1*z1+1e-12);den=np.maximum(alpha*d2+(1-alpha)*z1,1e-6);u=fx*x/den+cx;v=fy*y/den+cy
  result.append(((z>0)&(u>=0)&(u<1600)&(v>=0)&(v<1300)).sum(axis=1).tolist())
 return result
j0,v0=model(body);j1,v1=model(modified)
result={'perturbation':'right-multiply each local Head joint rotation by 5 degrees around local X','head_local_joint_body_pose_index':14,'frame_ids':fit['frame_ids'].tolist(),'observed_joint_max_displacement_m':float(np.linalg.norm(j1[:,obsidx]-j0[:,obsidx],axis=-1).max()),'head_joint_max_displacement_m':float(np.linalg.norm(j1[:,66]-j0[:,66],axis=-1).max()),'head_visible_counts_before_cam2_cam3':visible(v0),'head_visible_counts_after_cam2_cam3':visible(v1),'interpretation':'Image joint locations are insensitive to this local head rotation. Hard visibility counts are not the soft training loss; pose prior and soft visibility can still select a value.'}
with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
