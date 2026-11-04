"""Prepare unlabelled Sapiens2 observations and calibrated stereo with propagated covariance."""
from pathlib import Path
import json
import numpy as np
import argparse
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--detections',type=Path,required=True);a=p.parse_args();ROOT=a.output;ROOT.mkdir(parents=True,exist_ok=False)
DATA=a.detections

P={'score_min':0.3,'border_px':40,'pixel_sigma':8.,'max_ray_gap_m':.04,'min_range_m':.12,'max_range_m':2.5,'sigma_min_m':.008,'sigma_max_m':.5}
MAP={'left_elbow':'L_Elbow','right_elbow':'R_Elbow','left_wrist':'L_Wrist','right_wrist':'R_Wrist'}
for side in ['left','right']:
 for obs,smpl in [('thumb4','Thumb'),('forefinger4','Index'),('middle_finger4','Middle'),('ring_finger4','Ring'),('pinky_finger4','Pinky')]:
  MAP[side+'_'+obs]=('L_' if side=='left' else 'R_')+smpl+'_4'
NAMES=list(MAP); SMPL=list(MAP.values())
cal=json.loads(a.calibration.read_text());(ROOT/'camera_calibration.json').write_text(json.dumps(cal,indent=2)+'\n');R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);ds=[np.array(cal['cam2_ds']),np.array(cal['cam3_ds'])]
assert np.allclose(R.T@R,np.eye(3),atol=1e-5) and .04<np.linalg.norm(t)<.08

def unproject(uv,d):
 fx,fy,cx,cy,xi,a=d;mx=(uv[0]-cx)/fx;my=(uv[1]-cy)/fy;r2=mx*mx+my*my
 disc=1-(2*a-1)*r2
 if disc<=0:return None
 mz=(1-a*a*r2)/(a*np.sqrt(disc)+1-a); disc2=mz*mz+(1-xi*xi)*r2
 if disc2<=0:return None
 k=(mz*xi+np.sqrt(disc2))/(mz*mz+r2);v=np.array([k*mx,k*my,k*mz-xi]);return v/np.linalg.norm(v)

def triangulate(uv):
 a=unproject(uv[0],ds[0]); b=unproject(uv[1],ds[1])
 if a is None or b is None:return None
 b=R@b; sol=np.linalg.lstsq(np.stack([a,-b],axis=1),t,rcond=None)[0]
 p=a*sol[0];q=t+b*sol[1];gap=np.linalg.norm(p-q)
 return (p+q)*.5,sol,gap

with np.load(DATA,allow_pickle=False) as d:
 names=d['names'].tolist();frames=sorted(set(int(f) for f in d['frames']))
 assert len(frames)==180 and len(set(frames))==180
 index={(int(c),int(f)):i for i,(c,f) in enumerate(zip(d['cams'],d['frames']))}
 xy=np.zeros((len(frames),2,len(NAMES),2),np.float32);sc=np.zeros((len(frames),2,len(NAMES)),np.float32);valid=np.zeros_like(sc,dtype=bool)
 for fi,frame in enumerate(frames):
  for ci,cam in enumerate([2,3]):
   n=index[(cam,frame)]
   for j,name in enumerate(NAMES):
    k=names.index(name);xy[fi,ci,j]=d['xy'][n,k];sc[fi,ci,j]=d['score'][n,k];x,y=xy[fi,ci,j]
    valid[fi,ci,j]=np.isfinite([x,y,sc[fi,ci,j]]).all() and sc[fi,ci,j]>=P['score_min'] and P['border_px']<x<1600-P['border_px'] and P['border_px']<y<1300-P['border_px']
tri=np.zeros((len(frames),len(NAMES),3),np.float32);whiten=np.zeros((len(frames),len(NAMES),3,3),np.float32);tri_valid=np.zeros((len(frames),len(NAMES)),bool);gaps=np.full((len(frames),len(NAMES)),np.nan,np.float32)
for i in range(len(frames)):
 for j,name in enumerate(NAMES):
  if 'elbow' in name or 'wrist' not in name or not valid[i,:,j].all():continue
  uv=xy[i,:,j].astype(float);result=triangulate(uv)
  if result is None:continue
  pt,depth,gap=result;gaps[i,j]=gap
  if not np.isfinite(pt).all() or gap>P['max_ray_gap_m'] or np.min(depth)<P['min_range_m'] or np.max(depth)>P['max_range_m']:continue
  J=[]
  for k in range(4):
   p=uv.copy();m=uv.copy();p[k//2,k%2]+=.5;m[k//2,k%2]-=.5
   a,b=triangulate(p),triangulate(m)
   if a is None or b is None:break
   J.append(a[0]-b[0])
  if len(J)!=4:continue
  jac=np.stack(J,axis=1);cov=P['pixel_sigma']**2*jac@jac.T
  vals,vecs=np.linalg.eigh(cov);vals=np.clip(vals,P['sigma_min_m']**2,P['sigma_max_m']**2)
  tri[i,j]=pt;whiten[i,j]=(vecs*(1/np.sqrt(vals)))@vecs.T;tri_valid[i,j]=True
valid[:,:,4:]=False
payload=dict(frames=np.array(frames),names=np.array(NAMES),smpl_names=np.array(SMPL),xy=xy,score=sc,valid=valid,tri=tri,tri_valid=tri_valid,whiten=whiten,ray_gap_m=gaps)
with (ROOT/'observations.npz').open('xb') as f:np.savez_compressed(f,**payload)
with (ROOT/'observation_metadata.json').open('x') as f:json.dump(dict(settings=P,source='Sapiens2 existing detections; local asset',frame_count=len(frames),observation_counts={n:int(tri_valid[:,j].sum()) for j,n in enumerate(NAMES)},annotation_input=False),f,indent=2)
print('Prepared',len(frames),'frames. Triangulated points:',{n:int(tri_valid[:,j].sum()) for j,n in enumerate(NAMES)})
