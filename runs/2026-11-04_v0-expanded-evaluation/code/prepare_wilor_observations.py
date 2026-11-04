"""Add automatically matched WiLoR joints to fixed v0 observations, without labels."""
import argparse,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--wilor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
src=np.load(a.input/'observations.npz',allow_pickle=False);cal=json.loads((a.input/'camera_calibration.json').read_text());R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);ds=[cal['cam2_ds'],cal['cam3_ds']];frames=src['frames'];base_names=src['names'].tolist();smpl=src['smpl_names'].tolist();names=base_names[:4];smpl=smpl[:4];extra=[]
for side,short in [('left','L'),('right','R')]:
 for finger in ['Thumb','Index','Middle','Ring','Pinky']:
  for j in range(1,5):names.append(f'{side}_{finger.lower()}{j}');smpl.append(f'{short}_{finger}_{j}');extra.append((side,finger,j))
n=len(frames);N=len(names);xy=np.zeros((n,2,N,2),np.float32);score=np.zeros((n,2,N),np.float32);valid=np.zeros((n,2,N),bool);tri=np.zeros((n,N,3),np.float32);tv=np.zeros((n,N),bool);W=np.zeros((n,N,3,3),np.float32)
# Explicit indexing avoids confusing camera and joint axes.
for key,dest in [('xy',xy),('score',score),('valid',valid)]:dest[:,:,:4]=src[key][:,:,:4]
for key,dest in [('tri',tri),('tri_valid',tv),('whiten',W)]:dest[:,:4]=src[key][:,:4]
def unproj(uv,d):
 fx,fy,cx,cy,xi,al=d;x=(uv[0]-cx)/fx;y=(uv[1]-cy)/fy;r=x*x+y*y;disc=1-(2*al-1)*r
 if disc<=0:return None
 z=(1-al*al*r)/(al*np.sqrt(disc)+1-al);disc=z*z+(1-xi*xi)*r
 if disc<=0:return None
 k=(z*xi+np.sqrt(disc))/(z*z+r);v=np.array([k*x,k*y,k*z-xi]);return v/np.linalg.norm(v)
def triangulate(uv):
 av=unproj(uv[0],ds[0]);bv=unproj(uv[1],ds[1]);
 if av is None or bv is None:return None
 bv=R@bv;z=np.linalg.lstsq(np.stack([av,-bv],axis=1),t,rcond=None)[0];p=av*z[0];q=t+bv*z[1];return (p+q)/2,z,np.linalg.norm(p-q)
coverage=[]
for i,f in enumerate(frames):
 chosen={}
 for c,cam in enumerate((2,3)):
  path=a.wilor/'npz'/f'gen_{int(f):05d}_cam{cam}_wilor.npz'
  if not path.exists():continue
  with np.load(path,allow_pickle=False) as d:
   right=d['is_right'];points=d['joints_2d'];conf=d['detector_confidence']
   for side,side_num in [('left',0),('right',1)]:
    j=base_names.index(side+'_wrist');candidates=[]
    if not src['valid'][i,c,j]:continue
    for k in np.flatnonzero((right>0.5)==bool(side_num)):
     if not np.isfinite(points[k]).all():continue
     dist=float(np.linalg.norm(points[k,0]-src['xy'][i,c,j]));
     if dist<=100:candidates.append((dist,int(k)))
    if candidates:
     _,k=min(candidates);chosen[(c,side)]=(points[k].copy(),float(conf[k]))
 for j,(side,finger,level) in enumerate(extra,start=4):
  finger_i=['Thumb','Index','Middle','Ring','Pinky'].index(finger);wi=1+4*finger_i+level-1
  for c in (0,1):
   if (c,side) in chosen:
    pt,sc=chosen[(c,side)];u,v=pt[wi];xy[i,c,j]=pt[wi];score[i,c,j]=sc;valid[i,c,j]=np.isfinite([u,v,sc]).all() and 40<u<1560 and 40<v<1260
  if not valid[i,:,j].all():continue
  uv=xy[i,:,j].astype(float);out=triangulate(uv)
  if out is None:continue
  point,z,gap=out
  if not np.isfinite(point).all() or gap>.04 or np.min(z)<.12 or np.max(z)>2.5:continue
  deriv=[]
  for k in range(4):
   plus=uv.copy();minus=uv.copy();plus[k//2,k%2]+=.5;minus[k//2,k%2]-=.5;z1=triangulate(plus);z2=triangulate(minus)
   if z1 is None or z2 is None:break
   deriv.append(z1[0]-z2[0])
  if len(deriv)!=4:continue
  jac=np.stack(deriv,axis=1);eig,vec=np.linalg.eigh(64*jac@jac.T);eig=np.clip(eig,.008**2,.5**2);tri[i,j]=point;W[i,j]=(vec*(1/np.sqrt(eig)))@vec.T;tv[i,j]=True
 coverage.append({'frame':int(f),'matched_views':{s:[(c,s) in chosen for c in (0,1)] for s in ('left','right')},'hand_3d_points':{s:int(tv[i,[j for j,name in enumerate(names) if name.startswith(s+'_') and j>=4]].sum()) for s in ('left','right')}})
a.output.mkdir(parents=True,exist_ok=False);(a.output/'camera_calibration.json').write_text(json.dumps(cal,indent=2)+'\n');os.symlink((a.input/'initial').resolve(),a.output/'initial',target_is_directory=True)
with (a.output/'observations.npz').open('xb') as f:np.savez_compressed(f,frames=frames,names=np.array(names),smpl_names=np.array(smpl),xy=xy,score=score,valid=valid,tri=tri,tri_valid=tv,whiten=W)
(a.output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
