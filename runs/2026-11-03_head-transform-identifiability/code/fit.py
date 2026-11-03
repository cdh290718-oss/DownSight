"""Label-free multiframe SMPL-X stereo fitting with component ablations.

Only reads observations.npz, calibration, and unlabelled network predictions.
No DFM annotation is imported by this optimizer.
"""
import os,sys,json,time,argparse,copy
from pathlib import Path
import numpy as np, cv2, torch
from scipy.spatial.transform import Rotation
ROOT=Path(os.environ['STEREO_EXPERIMENT'])
REPO=Path(os.environ['EGOSMPLX_BACKEND'])
SOURCE=Path(os.environ['DOUBLE_SPHERE_SOURCE'])
sys.path.insert(0,str(SOURCE))
os.environ['EGOSMPLX_REPO_ROOT']=str(REPO)
os.environ['EGOSMPLX_CONFIG']=str(REPO/'main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py')
from egosmplx_pipeline.bootstrap import setup
smpl_x,_=setup(REPO)
from utils.smplx.smplx.lbs import batch_rodrigues
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=500);p.add_argument('--variant',choices=['full','no_shared','no_visibility','no_stereo'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--shoulders',action='store_true');p.add_argument('--weak-prior',action='store_true');p.add_argument('--stereo-weight',type=float,default=.5);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
weights=dict(reprojection=1.,stereo=a.stereo_weight,shared_position=.5,shared_rotation=.1,head_prior=(.0003 if a.weak_prior else 0.),visibility=.3,body_prior=.04,hand_prior=.01,orient_prior=.005,translation_prior=.001)
settings=dict(steps=a.steps,lr=.015,shared_learning_rate=.005,train_frames=10,variant=a.variant,weights=weights,head_center_prior_cam2_m=[0.,.1,-.02],head_center_prior_sigma_m=.18,anchor_position_sigma_m=.1,anchor_rotation_sigma=.25,robust_2d_scale_px=20.,uncertainty_sigma_px=8.,visibility_soft_border_px=20.,visibility_z_scale_m=.01,annotation_used=False,temporal_smoothness=False)
cal=json.loads((ROOT/'camera_calibration.json').read_text());obs=np.load(ROOT/'observations.npz',allow_pickle=False)
R=np.array(cal['stereo_R32'],np.float32);t32=np.array(cal['stereo_t32'],np.float32);virt=np.array(cal['cam3_R_virtual_from_gen'],np.float64)
frames=obs['frames'];N=len(frames);assert N==20
names=obs['smpl_names'].tolist();ji=[smpl_x.joints_name.index(n) for n in names]
head_ids=np.where(smpl_x.layer['neutral'].lbs_weights.detach().cpu().numpy().argmax(1)==smpl_x.orig_joints_name.index('Head'))[0];assert len(head_ids)==3345
layer=copy.deepcopy(smpl_x.layer['neutral']).cuda().eval();layer.requires_grad_(False)
T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
Tlong=lambda x:torch.as_tensor(x,dtype=torch.long,device='cuda')
raw=[]
for frame in frames:
 path=ROOT/'initial/egosmplx/smplx'/('gen_%05d_00_smplx.npz'%frame)
 with np.load(path,allow_pickle=False) as d:raw.append({k:d[k].copy() for k in d.files})
fixed_keys=['betas','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression']
fixed={k:T(np.stack([r[k].reshape(-1) for r in raw])) for k in fixed_keys}
body0=T(np.stack([r['body_pose'].reshape(21,3) for r in raw]));virtglob=np.stack([r['global_orient_network'].reshape(3) for r in raw]);virttrans=np.stack([r['transl_network'].reshape(3) for r in raw])
# SMPL-X rotates about pelvis. Transform both root rotation and translated pivot.
with torch.no_grad():
 pivot=layer(body_pose=body0.reshape(N,-1),global_orient=torch.zeros((N,3),device='cuda'),**fixed).joints[:,0].detach().cpu().numpy()
glob0=T(np.stack([cv2.Rodrigues(virt.T@cv2.Rodrigues(g)[0])[0].reshape(3) for g in virtglob]))
trans0=T(((virt.T@(pivot+virttrans).T).T-pivot).astype(np.float32))
cam2=T(cal['cam2_ds']);cam3=T(cal['cam3_ds']);stR=T(R);stt=T(t32);headidx=Tlong(head_ids);jidx=Tlong(ji)
def project(points,ds):
 fx,fy,cx,cy,xi,alpha=ds.unbind();x,y,z=points.unbind(-1);d1=torch.sqrt(x*x+y*y+z*z+1e-12);z1=xi*d1+z;d2=torch.sqrt(x*x+y*y+z1*z1+1e-12);den=alpha*d2+(1-alpha)*z1;den=den.clamp_min(1e-6)
 return torch.stack((fx*x/den+cx,fy*y/den+cy),-1)
def to3(x):return (x-stt)@stR
def head_rotation(glob,body):
 n=len(glob);rr=batch_rodrigues(glob.reshape(-1,3)).reshape(n,3,3);br=batch_rodrigues(body.reshape(-1,3)).reshape(n,21,3,3)
 for idx in [2,5,8,11,14]:rr=rr@br[:,idx]
 return rr
with torch.no_grad():
 Rheads=head_rotation(glob0,body0).cpu().numpy();rinit=Rotation.from_matrix(Rheads[:10]).mean().as_rotvec().astype(np.float32)
shared_prior=np.array([0.,.1,-.02],np.float32);
# Seven symmetric initializations, with independent rotation and centre perturbations.
directions=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],dtype=np.float32)
assert 0 <= a.seed < len(directions)
direction=directions[a.seed]
rinit=(Rotation.from_rotvec(direction*np.deg2rad(30))*Rotation.from_rotvec(rinit)).as_rotvec().astype(np.float32)
initial_center=shared_prior+direction*.15
tinit=-(Rotation.from_rotvec(rinit).as_matrix().T@initial_center).astype(np.float32)
settings.update(seed=a.seed,initial_center_cam2_m=initial_center.tolist(),initial_rotation_rotvec=rinit.tolist(),shoulders=a.shoulders,stereo_weight=a.stereo_weight)
if a.shoulders:
 with np.load(os.environ['SAPIENS_DETECTIONS'],allow_pickle=False) as det:
  idx={(int(c),int(f)):i for i,(c,f) in enumerate(zip(det['cams'],det['frames']))}
  k=[det['names'].tolist().index(x) for x in ['left_shoulder','right_shoulder','neck']]
  shoulder_xy=np.stack([[det['xy'][idx[(c,int(f))],k] for c in [2,3]] for f in frames])
  shoulder_sc=np.stack([[det['score'][idx[(c,int(f))],k] for c in [2,3]] for f in frames])
  shoulder_valid=np.isfinite(shoulder_xy).all(-1)&np.isfinite(shoulder_sc)&(shoulder_sc>=.3)&(shoulder_xy[...,0]>40)&(shoulder_xy[...,0]<1560)&(shoulder_xy[...,1]>40)&(shoulder_xy[...,1]<1260)
  settings['shoulder_valid_counts_train_test']=shoulder_valid[:10].sum(axis=(0,1)).tolist(),shoulder_valid[10:].sum(axis=(0,1)).tolist()

print('settings',json.dumps(settings),flush=True)

def fit(ids, shared_init, learn_shared):
 ids=np.array(ids);n=len(ids);dev=Tlong(ids);fo={k:v[dev] for k,v in fixed.items()};basebody=body0[dev];baseglob=glob0[dev];basetrans=trans0[dev]
 delta=torch.nn.Parameter(torch.zeros_like(basebody));gdelta=torch.nn.Parameter(torch.zeros_like(baseglob));trans=torch.nn.Parameter(basetrans.clone());lh=torch.nn.Parameter(fo['left_hand_pose'].clone());rh=torch.nn.Parameter(fo['right_hand_pose'].clone())
 sr=torch.nn.Parameter(T(shared_init[0]),requires_grad=learn_shared and a.variant!='no_shared');st=torch.nn.Parameter(T(shared_init[1]),requires_grad=learn_shared and a.variant!='no_shared')
 params=[{'params':[delta,gdelta,trans,lh,rh],'lr':settings['lr']}]
 if sr.requires_grad:params.append({'params':[sr,st],'lr':.005})
 opt=torch.optim.Adam(params)
 xy=T(obs['xy'][ids,0]);score=T(obs['score'][ids,0]);valid=torch.as_tensor(obs['valid'][ids,0],device='cuda');tri=T(obs['tri'][ids]);tri_valid=torch.as_tensor(obs['tri_valid'][ids],device='cuda');W=T(obs['whiten'][ids]);score2=T(obs['score'][ids].min(axis=1))
 assert valid.any()
 if a.shoulders:
  sxy=T(shoulder_xy[ids]); sscore=T(shoulder_sc[ids]); svalid=T(shoulder_valid[ids]); sjidx=Tlong([8,9,7])
 def forward():
  body=basebody+delta;glob=baseglob+gdelta
  f=dict(fo,left_hand_pose=lh,right_hand_pose=rh)
  out=layer(body_pose=body.reshape(n,-1),global_orient=glob,**f);v=out.vertices+trans[:,None,:];j=out.joints[:,smpl_x.joint_idx]+trans[:,None,:];head=out.joints[:,15]+trans;rot=head_rotation(glob,body)
  return v,j,head,rot
 def objective():
  v,j,head,rot=forward();uv=project(j[:,jidx],cam2);d=(uv-xy)/settings['robust_2d_scale_px'];two=(torch.log1p((d*d).sum(-1))*score.square()*valid).sum()/valid.sum()
  residue=torch.einsum('nkij,nkj->nki',W,j[:,jidx]-tri);tri_loss=(torch.log1p((residue*residue).sum(-1)/3)*score2.square()*tri_valid).sum()/tri_valid.sum().clamp_min(1)
  rs=batch_rodrigues(sr.reshape(1,3))[0];pred_camera=head+torch.einsum('nij,j->ni',rot,st);pos=(pred_camera/.1).square().sum(-1).mean();rotloss=((rot-rs)/.25).square().mean();center=-rs@st;prior=((center-T(shared_prior))/.18).square().mean()
  vis=v[:,headidx];vals=[]
  for points,ds in [(vis,cam2),(to3(vis),cam3)]:
   uvh=project(points,ds);u=uvh[...,0];vv=uvh[...,1];soft=torch.sigmoid(points[...,2]/.01)*torch.sigmoid(u/20)*torch.sigmoid((1600-u)/20)*torch.sigmoid(vv/20)*torch.sigmoid((1300-vv)/20);vals.append(soft.mean())
  visibility=torch.stack(vals).mean()
  bp=(delta/.7).square().mean();hp=((lh-fo['left_hand_pose'])/.7).square().mean()+((rh-fo['right_hand_pose'])/.7).square().mean();gp=(gdelta/.7).square().mean();tp=((trans-basetrans)/.5).square().mean()
  loss=weights['reprojection']*two+weights['body_prior']*bp+weights['hand_prior']*hp+weights['orient_prior']*gp+weights['translation_prior']*tp
  if a.shoulders:
   shoulder_j=j[:,sjidx]; projections=torch.stack([project(shoulder_j,cam2),project(to3(shoulder_j),cam3)],dim=1)
   residual=(projections-sxy)/20.; shoulder_loss=(torch.log1p((residual*residual).sum(-1))*sscore.square()*svalid).sum()/svalid.sum().clamp_min(1)
   loss=loss+shoulder_loss
  if a.variant!='no_stereo':loss=loss+weights['stereo']*tri_loss
  if a.variant!='no_shared':loss=loss+weights['shared_position']*pos+weights['shared_rotation']*rotloss+weights['head_prior']*prior
  if a.variant!='no_visibility':loss=loss+weights['visibility']*visibility
  return loss,dict(reprojection=two,stereo=tri_loss,shared_position=pos,shared_rotation=rotloss,head_prior=prior,visibility=visibility,body_prior=bp,hand_prior=hp), (v,j,head,rot)
 for step in range(a.steps):
  opt.zero_grad();loss,parts,_=objective()
  if not torch.isfinite(loss):raise RuntimeError('nonfinite loss at step %d'%step)
  loss.backward();torch.nn.utils.clip_grad_norm_([delta,gdelta,trans,lh,rh]+([sr,st] if sr.requires_grad else []),10.);opt.step()
  if step%100==0 or step==a.steps-1:print('frames',ids[0],ids[-1],'step',step,'loss',float(loss),'parts',{k:round(float(v.detach()),4) for k,v in parts.items()},flush=True)
 with torch.no_grad():
  loss,parts,(v,j,head,rot)=objective();rshared=batch_rodrigues(sr.reshape(1,3))[0];vis=v[:,headidx]
  counts=[]
  for points,ds in [(vis,cam2),(to3(vis),cam3)]:
   uv=project(points,ds);good=(points[...,2]>0)&(uv[...,0]>=0)&(uv[...,0]<1600)&(uv[...,1]>=0)&(uv[...,1]<1300);counts.append(good.sum(1))
  cpu=lambda x:x.detach().cpu().numpy()
  result=dict(frame_ids=frames[ids],joints_cam2=cpu(j),head_joint_cam2=cpu(head),head_R_cam2=cpu(rot),vertices_cam2=cpu(v),body_pose=cpu(basebody+delta),global_orient=cpu(baseglob+gdelta),transl=cpu(trans),left_hand_pose=cpu(lh),right_hand_pose=cpu(rh),shared_rotation_head_to_cam2=cpu(rshared),shared_camera_in_head_m=cpu(st),shared_head_center_cam2_m=cpu(-rshared@st),visible_head_count=np.stack([cpu(x) for x in counts],axis=1),final_loss=np.array(float(loss)),triangulated_count=obs['tri_valid'][ids].sum(axis=1))
  return result,(cpu(sr),cpu(st))
start=time.time();train,shared=fit(range(10),(rinit,tinit),True);test,_=fit(range(10,20),shared,False)
for name,obj in [('train',train),('test',test)]:
 with (a.output/(name+'.npz')).open('xb') as f:np.savez_compressed(f,**obj)
with (a.output/'settings.json').open('x') as f:json.dump(dict(settings,elapsed_seconds=time.time()-start,shared_head_to_cam2_rotation=train['shared_rotation_head_to_cam2'].tolist(),shared_camera_in_head_m=train['shared_camera_in_head_m'].tolist(),shared_head_center_cam2_m=train['shared_head_center_cam2_m'].tolist()),f,indent=2)
print('DONE',a.variant,'seconds',time.time()-start,flush=True)
