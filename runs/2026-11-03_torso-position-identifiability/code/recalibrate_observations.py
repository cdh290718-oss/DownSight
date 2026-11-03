"""Use train-image relative pose and externally supplied baseline, never DFM body labels."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source-calibration',type=Path,required=True);p.add_argument('--feature-estimate',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=float,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);cal=json.loads(a.source_calibration.read_text());est=json.loads(a.feature_estimate.read_text());R21=np.array(est['R_cam2_to_cam3']);t21=np.array(est['translation_direction_cam3'])*a.baseline;R=R21.T;t=-R@t21;cal['stereo_R32']=R.tolist();cal['stereo_t32']=t.tolist();cal['calibration_note']='Relative direction and rotation estimated using training images; translation scale from user-specified baseline.'
with (a.output/'camera_calibration.json').open('x') as f:json.dump(cal,f,indent=2);f.write('\n')
source=np.load(a.observations,allow_pickle=False);ds=[np.array(cal['cam2_ds']),np.array(cal['cam3_ds'])]
def unproject(uv,d):
 fx,fy,cx,cy,xi,alpha=d;mx=(uv[0]-cx)/fx;my=(uv[1]-cy)/fy;r2=mx*mx+my*my;disc=1-(2*alpha-1)*r2
 if disc<=0:return None
 mz=(1-alpha*alpha*r2)/(alpha*np.sqrt(disc)+1-alpha);disc2=mz*mz+(1-xi*xi)*r2
 if disc2<=0:return None
 k=(mz*xi+np.sqrt(disc2))/(mz*mz+r2);v=np.array([k*mx,k*my,k*mz-xi]);return v/np.linalg.norm(v)
def triangulate(uv):
 av=unproject(uv[0],ds[0]);bv=unproject(uv[1],ds[1])
 if av is None or bv is None:return None
 bv=R@bv;sol=np.linalg.lstsq(np.stack([av,-bv],axis=1),t,rcond=None)[0];p=av*sol[0];q=t+bv*sol[1];return (p+q)*.5,sol,np.linalg.norm(p-q)
frames=source['frames']; names=source['names'].tolist();xy=source['xy'];valid=source['valid'];tri=np.zeros((len(frames),len(names),3),np.float32);W=np.zeros((len(frames),len(names),3,3),np.float32);tv=np.zeros((len(frames),len(names)),bool);gaps=np.full((len(frames),len(names)),np.nan,np.float32)
for i in range(len(frames)):
 for j,name in enumerate(names):
  if 'elbow' in name or not valid[i,:,j].all():continue
  uv=xy[i,:,j].astype(float);out=triangulate(uv)
  if out is None:continue
  point,depth,gap=out;gaps[i,j]=gap
  if not np.isfinite(point).all() or gap>.04 or np.min(depth)<.12 or np.max(depth)>2.5:continue
  J=[]
  for k in range(4):
   plus=uv.copy();minus=uv.copy();plus[k//2,k%2]+=.5;minus[k//2,k%2]-=.5;v1=triangulate(plus);v2=triangulate(minus)
   if v1 is None or v2 is None:break
   J.append(v1[0]-v2[0])
  if len(J)!=4:continue
  jac=np.stack(J,axis=1);eig,vec=np.linalg.eigh(64.*jac@jac.T);eig=np.clip(eig,.008**2,.5**2);tri[i,j]=point;W[i,j]=(vec*(1/np.sqrt(eig)))@vec.T;tv[i,j]=True
payload={k:source[k] for k in ('frames','names','smpl_names','xy','score','valid')};payload.update(tri=tri,whiten=W,tri_valid=tv,ray_gap_m=gaps)
with (a.output/'observations.npz').open('xb') as f:np.savez_compressed(f,**payload)
print('train-test triangulated wrist counts', {name:[int(tv[:10,names.index(name)].sum()),int(tv[10:,names.index(name)].sum())] for name in ('left_wrist','right_wrist')})
