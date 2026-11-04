"""Frozen-body, wrist-center-preserving WiLoR hand stage; reads no DFM labels."""
import argparse,copy,json,os,sys,time
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--v0',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--first',type=int,required=True);p.add_argument('--last',type=int,required=True);a=p.parse_args()
assert 0<=a.first<a.last<=180 and a.first%10==0 and a.last%10==0
repo=Path(os.environ['EGOSMPLX_BACKEND']);source=Path(os.environ['DOUBLE_SPHERE_SOURCE']);sys.path.insert(0,str(source));os.environ['EGOSMPLX_REPO_ROOT']=str(repo);os.environ['EGOSMPLX_CONFIG']=str(repo/'main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py')
from egosmplx_pipeline.bootstrap import setup
smpl_x,_=setup(repo);assert smpl_x.orig_joints_name[20:22]==('L_Wrist','R_Wrist')
layer=copy.deepcopy(smpl_x.layer['neutral']).cuda().eval();layer.requires_grad_(False);T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');obs=np.load(a.observations,allow_pickle=False);cal=json.loads(a.calibration.read_text());cam2=T(cal['cam2_ds']);names=obs['smpl_names'].tolist();jidx=torch.tensor([smpl_x.joints_name.index(n) for n in names[4:]],device='cuda');assert len(jidx)==40
idx={int(f):i for i,f in enumerate(obs['frames'])};a.output.mkdir(parents=True,exist_ok=False);start=time.time();torch.manual_seed(0)
def project(x,ds):
 fx,fy,cx,cy,xi,alpha=ds.unbind();X,Y,Z=x.unbind(-1);d1=torch.sqrt(X*X+Y*Y+Z*Z+1e-12);z1=xi*d1+Z;den=(alpha*torch.sqrt(X*X+Y*Y+z1*z1+1e-12)+(1-alpha)*z1).clamp_min(1e-6);return torch.stack([fx*X/den+cx,fy*Y/den+cy],-1)
for begin in range(a.first,a.last,10):
 with np.load(a.v0/f'batch_{begin:03d}.npz',allow_pickle=False) as d:base={k:d[k].copy() for k in d.files}
 frames=base['frame_ids'];n=len(frames);ids=[idx[int(f)] for f in frames];raw=[]
 for f in frames:
  with np.load(a.initial/'smplx'/f'gen_{int(f):05d}_00_smplx.npz',allow_pickle=False) as d:raw.append({k:d[k].copy() for k in ('betas','jaw_pose','leye_pose','reye_pose','expression')})
 fixed={k:T(np.stack([r[k].reshape(-1) for r in raw])) for k in raw[0]};body=T(base['body_pose']);glob=T(base['global_orient']);trans=T(base['transl']);lh0=T(base['left_hand_pose']);rh0=T(base['right_hand_pose']);wd=torch.nn.Parameter(torch.zeros((n,2,3),device='cuda'));lh=torch.nn.Parameter(lh0.clone());rh=torch.nn.Parameter(rh0.clone());opt=torch.optim.Adam([wd,lh,rh],lr=.015)
 xy=T(obs['xy'][ids,0,4:]).reshape(n,2,20,2);score=T(obs['score'][ids,0,4:]).reshape(n,2,20);valid=torch.as_tensor(obs['valid'][ids,0,4:].reshape(n,2,20),device='cuda');tri=T(obs['tri'][ids,4:]).reshape(n,2,20,3);tv=torch.as_tensor(obs['tri_valid'][ids,4:].reshape(n,2,20),device='cuda');W=T(obs['whiten'][ids,4:]).reshape(n,2,20,3,3);score2=T(obs['score'][ids,:,4:].min(axis=1)).reshape(n,2,20);active=(valid.any(-1)|tv.any(-1)).float()
 def forward():
  effective_wd=wd*active[...,None];effective_lh=lh0+(lh-lh0)*active[:,0,None];effective_rh=rh0+(rh-rh0)*active[:,1,None];newbody=torch.cat((body[:,:19,:],body[:,19:,:]+effective_wd),dim=1);out=layer(body_pose=newbody.reshape(n,-1),global_orient=glob,transl=trans,left_hand_pose=effective_lh,right_hand_pose=effective_rh,**fixed);j=out.joints[:,smpl_x.joint_idx];return j,newbody
 with torch.no_grad():
  initial,_=forward();err=np.max(np.abs(initial.cpu().numpy()[:,:65]-base['joints_cam2'][:,:65]));assert err<1e-5,('v0 reconstruction mismatch',begin,float(err))
 for step in range(1500):
  opt.zero_grad();j,_=forward();hands=j[:,jidx].reshape(n,2,20,3);uv=project(hands,cam2);d=(uv-xy)/20;two=(torch.log1p((d*d).sum(-1))*score.square()*valid).sum(-1)/valid.sum(-1).clamp_min(1);res=torch.einsum('nshij,nshj->nshi',W,hands-tri);three=(torch.log1p((res*res).sum(-1)/3)*score2.square()*tv).sum(-1)/tv.sum(-1).clamp_min(1);finger=torch.stack([((lh-lh0)/.7).square().mean(-1),((rh-rh0)/.7).square().mean(-1)],dim=1);wprior=(wd/.7).square().mean(-1);loss=(two+.5*three+.04*wprior+.01*finger).sum(-1).mean()
  if not torch.isfinite(loss):raise RuntimeError(f'nonfinite batch {begin} step {step}')
  loss.backward();torch.nn.utils.clip_grad_norm_([wd,lh,rh],10.);opt.step()
  if step%500==0 or step==1499:print('batch',begin,'step',step,'loss',float(loss),flush=True)
 with torch.no_grad():
  j,newbody=forward();out=j.cpu().numpy();arm_diff=np.linalg.norm(out[:,[10,11,12,13]]-base['joints_cam2'][:,[10,11,12,13]],axis=-1);assert np.max(arm_diff)<1e-5,(begin,float(np.max(arm_diff)));noobs=~(valid.any(-1)|tv.any(-1));change=np.linalg.norm((wd*active[...,None]).detach().cpu().numpy(),axis=-1);assert np.all(change[noobs.cpu().numpy()]<1e-9);assert torch.equal((lh0+(lh-lh0)*active[:,0,None])[noobs[:,0]],lh0[noobs[:,0]]) and torch.equal((rh0+(rh-rh0)*active[:,1,None])[noobs[:,1]],rh0[noobs[:,1]])
  result=dict(frame_ids=frames,joints_cam2=out,body_pose=newbody.cpu().numpy(),left_hand_pose=(lh0+(lh-lh0)*active[:,0,None]).cpu().numpy(),right_hand_pose=(rh0+(rh-rh0)*active[:,1,None]).cpu().numpy(),wrist_rotation_delta=(wd*active[...,None]).cpu().numpy(),max_arm_diff_m=np.array(float(np.max(arm_diff))),initial_reconstruction_max_m=np.array(float(err)),no_observations=noobs.cpu().numpy())
  with (a.output/f'batch_{begin:03d}.npz').open('xb') as f:np.savez_compressed(f,**result)
 print('done batch',begin,'arm_max_m',float(np.max(arm_diff)),'missing_hand_count',int(noobs.sum()),flush=True)
(a.output/'settings.json').write_text(json.dumps({'steps':1500,'lr':.015,'stereo_weight':.5,'hand_prior':.01,'wrist_prior':.04,'sigma_rad':.7,'source':'v0 fixed body; WiLoR matched observations','seconds':time.time()-start},indent=2)+'\n')
