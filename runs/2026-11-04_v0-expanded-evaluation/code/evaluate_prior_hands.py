"""Post-fit 10-frame hand diagnostic; references never enter optimization."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ref=np.load(a.reference);lookup={int(f):i for i,f in enumerate(ref['frames'])};out={'scope':'10 previously inspected development frames; DFM is model reference','variants':{}}
def frames(x):
 a=x[:,5]-x[:,0];b=x[:,17]-x[:,0];a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12);z=np.cross(a,b);z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12);y=np.cross(z,a);return np.stack([a,y,z],axis=-1)
for mode in ['baseline','wrist_only']:
 d=np.load(a.fits/f'{mode}_1500/test.npz');ix=[lookup[int(f)] for f in d['frame_ids']];entry={}
 for si,side in enumerate(['left','right']):
  ids=[12 if si==0 else 13]+list(range(25 if si==0 else 45,45 if si==0 else 65)); pred=d['joints_cam2'][:,ids];gt=ref['cam2_hand_xyz'][ix,si];rel=(pred-pred[:,:1])-(gt-gt[:,:1]);err=np.linalg.norm(rel,axis=-1).mean(axis=-1)*100;Rp=frames(pred);Rg=frames(gt);cos=np.clip((np.einsum('nij,nij->n',Rp,Rg)-1)/2,-1,1);angle=np.degrees(np.arccos(cos));ok=np.isfinite(err)&np.isfinite(angle);entry[side]={'count':int(ok.sum()),'relative_21_point_mean_error_cm_median':float(np.median(err[ok])),'relative_21_point_mean_error_cm_p90':float(np.percentile(err[ok],90)),'palm_frame_rotation_error_deg_median':float(np.median(angle[ok])),'palm_frame_rotation_error_deg_p90':float(np.percentile(angle[ok],90)),'per_frame':[{'frame':int(f),'mean_error_cm':float(e),'rotation_error_deg':float(g)} for f,e,g,o in zip(d['frame_ids'],err,angle,ok) if o]}
 out['variants'][mode]=entry
with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
