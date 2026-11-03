"""SMPL-X stereo-point simulation; fixed shape/vertex correspondences are optimistic oracles."""
import os,sys,json,argparse,copy,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
repo=Path(os.environ['EGOSMPLX_BACKEND']);source=Path(os.environ['DOUBLE_SPHERE_SOURCE']);sys.path.insert(0,str(source));os.environ['EGOSMPLX_REPO_ROOT']=str(repo);os.environ['EGOSMPLX_CONFIG']=str(repo/'main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py')
from egosmplx_pipeline.bootstrap import setup
smpl,_=setup(repo)
p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True);p.add_argument('--fit',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--rectification',type=Path,required=True);p.add_argument('--axes',type=int,choices=[1,2],required=True);p.add_argument('--degrees',type=float,required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--steps',type=int,default=600);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
N=12;S=7;K=128;rng=np.random.default_rng(20261103+a.trial);htrue=np.array([0.,.12,-.04],dtype=np.float32);init=np.load(a.initial,allow_pickle=False);basefit=np.load(a.fit,allow_pickle=False);body0=basefit['body_pose'][0].reshape(21,3).copy();g0=basefit['global_orient'][0].copy();cal=json.loads(a.calibration.read_text());rect=np.load(a.rectification,allow_pickle=False);R1=rect['rectification_R1'];P1=rect['P1'];P2=rect['P2'];R32=np.array(cal['stereo_R32']);t32=np.array(cal['stereo_t32']);ds=np.array(cal['cam2_ds']);ds3=np.array(cal['cam3_ds']);layer=copy.deepcopy(smpl.layer['neutral']).cuda().eval();layer.requires_grad_(False);T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');key=['betas','jaw_pose','leye_pose','reye_pose','expression','left_hand_pose','right_hand_pose'];fixed_np={k:init[k].reshape(-1) for k in key};fixed_np['left_hand_pose']=basefit['left_hand_pose'][0];fixed_np['right_hand_pose']=basefit['right_hand_pose'][0]
def fixed(n):return {k:T(np.broadcast_to(v,(n,len(v))).copy()) for k,v in fixed_np.items()}
def chain(b):
 q=Rotation.identity()
 for i in (2,5,8,11,14):q=q*Rotation.from_rotvec(b[i])
 return q
head0=Rotation.from_rotvec(g0)*chain(body0);phase=np.linspace(0,2*np.pi,N,endpoint=False); bodies=[];globs=[];motion_angles=[]
for i in range(N):
 rv=np.array([np.sin(phase[i]),0.,(np.sin(2*phase[i]) if a.axes==2 else 0.)])*np.deg2rad(a.degrees);motion_angles.append(float(np.degrees(np.linalg.norm(rv))));b=body0.copy();b[11]=(Rotation.from_rotvec(body0[11])*Rotation.from_rotvec(rv)).as_rotvec();g=(head0*chain(b).inv()).as_rotvec();bodies.append(b);globs.append(g)
bodies=np.array(bodies,dtype=np.float32);globs=np.array(globs,dtype=np.float32)
with torch.no_grad():
 out=layer(body_pose=T(bodies.reshape(N,-1)),global_orient=T(globs),**fixed(N));verts=(out.vertices+(T(htrue)[None,:]-out.joints[:,15])[:,None,:]).cpu().numpy()
lbs=smpl.layer['neutral'].lbs_weights.detach().cpu().numpy();regions=lbs.argmax(1);choices=np.flatnonzero(np.isin(regions,[3,6,9,12,13,14,16,17,18,19,20,21]));indices=[];obs=[];sigmas=[]
def project(v,d):
 fx,fy,cx,cy,xi,alpha=d;x,y,z=v.T;d1=np.sqrt(x*x+y*y+z*z);z1=xi*d1+z;d2=np.sqrt(x*x+y*y+z1*z1);den=alpha*d2+(1-alpha)*z1;return np.c_[fx*x/np.maximum(den,1e-8)+cx,fy*y/np.maximum(den,1e-8)+cy]
for i in range(N):
 xyz=verts[i,choices];q=xyz@R1.T;uv=project(xyz,ds);xyz3=(xyz-t32)@R32;uv3=project(xyz3,ds3);u=P1[0,0]*q[:,0]/q[:,2]+P1[0,2];v=P1[1,1]*q[:,1]/q[:,2]+P1[1,2];u3=u+P2[0,3]/q[:,2];valid=(q[:,2]>.12)&(np.linalg.norm(xyz,axis=1)<1.2)&(uv[:,0]>10)&(uv[:,0]<1590)&(uv[:,1]>10)&(uv[:,1]<1290)&(uv3[:,0]>10)&(uv3[:,0]<1590)&(uv3[:,1]>10)&(uv3[:,1]<1290)&(u>5)&(u<795)&(u3>5)&(u3<795)&(v>5)&(v<645)
 candidates=np.flatnonzero(valid)
 if len(candidates)<K:raise RuntimeError('insufficient projected visible candidates '+str((i,len(candidates))))
 pick=rng.choice(candidates,size=K,replace=False);indices.append(choices[pick]);q=q[pick];u=u[pick];v=v[pick];u3=u3[pick]
 # Independent 1 px Gaussian noise in each camera; rectified triangulation.
 uL=u+rng.normal(0,1,K);uR=u3+rng.normal(0,1,K);vL=v+rng.normal(0,1,K);vR=v+rng.normal(0,1,K);disp=uL-uR;z=-P2[0,3]/disp;xp=(uL-P1[0,2])*z/P1[0,0];yp=((vL+vR)*.5-P1[1,2])*z/P1[1,1];obs.append(np.c_[xp,yp,z]@R1);sigmas.append(np.c_[np.full(K,.005),np.full(K,.005),np.maximum(.003,np.sqrt(2.)*z*z/abs(P2[0,3]))])
indices=np.array(indices);obs=np.array(obs,np.float32);sigmas=np.array(sigmas,np.float32);starts=np.vstack([np.zeros(3),np.eye(3),-np.eye(3)]).astype(np.float32)*.15;h=torch.nn.Parameter(T(htrue+starts));delta=torch.nn.Parameter(torch.zeros((S,N,21,3),device='cuda'));gd=torch.nn.Parameter(torch.zeros((S,N,3),device='cuda'));opt=torch.optim.Adam([h,delta,gd],lr=.01)
# Ground truth torso orientation is used as initialisation (optimistic); motion of the neck is not supplied.
bodyinit=np.broadcast_to(body0,(S,N,21,3)).copy();globinit=np.broadcast_to(globs,(S,N,3)).copy();base=T(bodyinit);glob=T(globinit);index=torch.as_tensor(np.broadcast_to(indices,(S,N,K)).copy().reshape(S*N,K),dtype=torch.long,device='cuda');target=T(np.broadcast_to(obs,(S,N,K,3)).copy());sigma=T(np.broadcast_to(sigmas,(S,N,K,3)).copy());fixed_batch=fixed(S*N)
start=time.time()
for step in range(a.steps):
 opt.zero_grad();o=layer(body_pose=(base+delta).reshape(S*N,-1),global_orient=(glob+gd).reshape(S*N,3),**fixed_batch);sel=torch.gather(o.vertices,1,index[:,:,None].expand(-1,-1,3));head=o.joints[:,15];pred=(sel+(h[:,None,:].expand(S,N,3).reshape(S*N,3)-head)[:,None,:]).reshape(S,N,K,3);res=(pred-target)/sigma
 per=(torch.sqrt(1+(res*res).sum(-1)/9)-1).mean(dim=(1,2));prior=(delta*delta).mean(dim=(1,2,3))*1e-4+(gd*gd).mean(dim=(1,2))*1e-4;loss=(per+prior).sum();loss.backward();torch.nn.utils.clip_grad_norm_([h,delta,gd],100);opt.step()
 if step%200==0 or step==a.steps-1:print('step',step,'losses',per.detach().cpu().tolist(),flush=True)
with torch.no_grad():
 centers=h.detach().cpu().numpy();err=np.linalg.norm(centers-htrue,axis=1)*100;spread=max(np.linalg.norm(x-y) for x in centers for y in centers)*100
result={'axes':a.axes,'amplitude_deg':a.degrees,'trial':a.trial,'steps':a.steps,'frame_count':N,'points_per_frame':K,'max_neck_rotation_deg':max(motion_angles),'head_camera_rotation_max_error_deg':float(max(np.degrees((head0.inv()*(Rotation.from_rotvec(g)*chain(b))).magnitude()) for g,b in zip(globs,bodies))),'pixel_noise_std_px':1.,'centers_m':centers.tolist(),'true_center_m':htrue.tolist(),'error_cm':err.tolist(),'max_pairwise_cm':float(spread),'final_per_start_objective':per.detach().cpu().tolist(),'synthetic_conditions':'SMPL-X exact shape; identity correspondences; projected FOV only, no occlusion or clothing; true torso orientation used only as initial estimate; no DFM','elapsed_seconds':time.time()-start}
with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
