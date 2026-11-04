"""Frozen-protocol post-fit evaluation; DFM reference is used only here."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=np.load(a.reference,allow_pickle=False);frames=r['frames'];n=len(frames);assert n==180;index={int(f):i for i,f in enumerate(frames)};previous=set(range(0,5131,270));rng=np.random.default_rng(0)
def interval(x,kind,miss=None):
 x=np.asarray(x,float);valid=np.isfinite(x)
 if not valid.any():return None
 val=lambda arr: float(np.median(arr)) if kind=='median' else float(np.percentile(arr,90)) if kind=='p90' else float(np.mean(arr>5))
 result=val(x[valid]);n=len(x);boot=[]
 for _ in range(2000):
  start=rng.integers(0,n,size=int(np.ceil(n/10)));ii=(start[:,None]+np.arange(10))%n;arr=x[ii.ravel()[:n]];arr=arr[np.isfinite(arr)]
  if len(arr):boot.append(val(arr))
 return {'value':result,'ci95':np.percentile(boot,[2.5,97.5]).tolist()}
def palm(x):
 a=x[:,5]-x[:,0];b=x[:,17]-x[:,0];norm=np.linalg.norm(a,axis=1);z=np.cross(a,b);normz=np.linalg.norm(z,axis=1);ok=(norm>1e-6)&(normz>1e-8);a=a/np.maximum(norm[:,None],1e-12);z=z/np.maximum(normz[:,None],1e-12);return np.stack([a,np.cross(z,a),z],axis=-1),ok
out={'note':'DFM model reference, not independent truth; 180 within one sequence. The previous 20 frames were used in development; 10 are also used in simulated calibration.','frames':frames.tolist(),'variants':{},'per_frame':{}}
for mode in ['v0','wilor']:
 data={};present=np.zeros(n,bool)
 for path in sorted((a.fits/mode).glob('batch_*.npz')):
  with np.load(path,allow_pickle=False) as d:
   for k,f in enumerate(d['frame_ids']):
    i=index[int(f)];assert not present[i];present[i]=True;data[i]=d['joints_cam2'][k].copy()
 predicted=np.full((n,65,3),np.nan)
 for i,j in data.items():
  assert j.shape[0]>=65 and j.shape[1]==3
  predicted[i]=j[:65]
 vals={};records=[];hand={}
 for side,side_i,wi,ei,gtw,gte,start in [('left',0,12,10,20,18,25),('right',1,13,11,21,19,45)]:
  for joint,ji,gt in [('wrist',wi,gtw),('elbow',ei,gte)]:
   truth=r['cam2_body_xyz'][:,gt];refok=r['cam2_body_valid'][:,gt]&np.isfinite(truth).all(-1);err=np.linalg.norm(predicted[:,ji]-truth,axis=1)*100;err[~refok]=np.nan;failure=np.where(refok,np.where(np.isfinite(err),err>5,True),False)
   result={}
   for label,subset in [('all',np.ones(n,bool)),('new160',np.array([int(f) not in previous for f in frames]))]:
    eligible=refok&subset;vec=np.where(eligible,err,np.nan);num=int(eligible.sum());failure_vec=np.where(eligible,np.where(np.isfinite(err),err,1000000),np.nan);result[label]={'reference_valid':num,'prediction_valid':int((eligible&np.isfinite(err)).sum()),'median_cm':interval(vec,'median'),'p90_cm':interval(vec,'p90'),'failure_fraction_gt5_or_missing':interval(failure_vec,'failure'),'missing_predictions':int((eligible&~np.isfinite(err)).sum())}
   vals[f'{side}_{joint}']=result
  ids=[wi]+list(range(start,start+20));pred=predicted[:,ids];truth=r['cam2_hand_xyz'][:,side_i];rel=(pred-pred[:,:1])-(truth-truth[:,:1]);e=np.linalg.norm(rel,axis=2).mean(axis=1)*100;Rp,okp=palm(pred);Rt,okt=palm(truth);cos=np.clip((np.einsum('nij,nij->n',Rp,Rt)-1)/2,-1,1);angle=np.degrees(np.arccos(cos));angle[~(okp&okt)]=np.nan
  hand[side]={}
  for label,subset in [('all',np.ones(n,bool)),('new160',np.array([int(f) not in previous for f in frames]))]:
   x=np.where(subset,e,np.nan);y=np.where(subset,angle,np.nan);hand[side][label]={'valid_21_point_count':int(np.isfinite(x).sum()),'mean_relative_21_point_error_cm_median':interval(x,'median'),'mean_relative_21_point_error_cm_p90':interval(x,'p90'),'valid_palm_count':int(np.isfinite(y).sum()),'palm_rotation_error_deg_median':interval(y,'median'),'palm_rotation_error_deg_p90':interval(y,'p90')}
  if side=='left':left_hand_error=e;left_angle=angle
  else:right_hand_error=e;right_angle=angle
  for i,f in enumerate(frames):
   if side=='left':records.append({'frame':int(f),'sides':{}})
   uv=np.stack([r['cam2_body_uv'][i,gtw],r['cam3_body_uv'][i,gtw]]);outside=bool(np.any(~np.isfinite(uv)) or np.any((uv[:,0]<0)|(uv[:,0]>=1600)|(uv[:,1]<0)|(uv[:,1]>=1300)))
   close=bool(r['cam2_body_valid'][i,[20,21]].all() and np.isfinite(r['cam2_body_xyz'][i,[20,21]]).all() and np.linalg.norm(r['cam2_body_xyz'][i,20]-r['cam2_body_xyz'][i,21])<.1)
   hh=r['cam2_hand_xyz'][i];min_point_dist=np.min(np.linalg.norm(hh[0,:,None,:]-hh[1,None,:,:],axis=-1)) if np.isfinite(hh).all() else np.nan
   ee=np.linalg.norm(predicted[i,wi]-r['cam2_body_xyz'][i,gtw])*100
   records[i]['sides'][side]={'dfm_valid':bool(r['cam2_body_valid'][i,gtw]),'prediction_present':bool(present[i]),'wrist_error_cm':float(ee) if np.isfinite(ee) else None,'either_view_wrist_outside':outside,'hands_within_10cm':close,'hands_21_points_within_5cm_exploratory':bool(np.isfinite(min_point_dist) and min_point_dist<.05)}
 out['variants'][mode]={'body':vals,'hand':hand,'prediction_count':int(present.sum())};out['per_frame'][mode]=records
with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
