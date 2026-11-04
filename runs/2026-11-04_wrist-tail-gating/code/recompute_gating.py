import argparse, json, csv, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from PIL import Image, ImageDraw

p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True);p.add_argument('--cal',type=Path,required=True);p.add_argument('--out',type=Path,required=True); a=p.parse_args()
a.out.mkdir(exist_ok=False,parents=True)
d=np.load(a.data/'det_sapiens2.npz'); cal=json.loads(a.cal.read_text()); ds=[np.array(cal['cam2_ds']),np.array(cal['cam3_ds'])]; R=np.array(cal['stereo_R32']); t=np.array(cal['stereo_t32']); names=d['names'].tolist(); select=names[5:63]; frames=list(range(0,5131,270)); lookup={(int(c),int(f)):i for i,(c,f) in enumerate(zip(d['cams'],d['frames']))}

def unproj(uv,d):
 fx,fy,cx,cy,xi,al=d; x=(uv[0]-cx)/fx;y=(uv[1]-cy)/fy;r=x*x+y*y; z=(1-al*al*r)/(al*np.sqrt(max(0,1-(2*al-1)*r))+1-al); k=(z*xi+np.sqrt(z*z+(1-xi*xi)*r))/(z*z+r);v=np.array([k*x,k*y,k*z-xi]);return v/np.linalg.norm(v)
def proj(x,d):
 fx,fy,cx,cy,xi,al=d; n=np.linalg.norm(x); z=xi*n+x[2]; den=al*np.sqrt(x[0]**2+x[1]**2+z*z)+(1-al)*z;return np.array([fx*x[0]/den+cx,fy*x[1]/den+cy])
def project(x):return np.array([proj(x,ds[0]),proj(R.T@(x-t),ds[1])])
def tri(uv):
 v=unproj(uv[0],ds[0]);w=R@unproj(uv[1],ds[1]); z=np.linalg.lstsq(np.stack([v,-w],axis=1),t,rcond=None)[0];return (v*z[0]+t+w*z[1])/2,z
rows=[]; points={}; images={}; base_names=[s+'_'+n for s in ('left','right') for n in ('elbow','wrist','thumb4','forefinger4','middle_finger4','ring_finger4','pinky_finger4')]
for f in frames:
 xy=np.array([d['xy'][lookup[c,f]] for c in (2,3)]); sc=np.array([d['score'][lookup[c,f]] for c in (2,3)]); temp={}
 for name in select:
  j=names.index(name); uv=xy[:,j].astype(float); scores=sc[:,j]; border=np.minimum.reduce([uv[:,0],1600-uv[:,0],uv[:,1],1300-uv[:,1]]); old=(np.array([name in base_names]*2)&(scores>=.3)&(border>40)); reason=[]; corrected=None; xyz=None; cov=None; epi=None; depth=None; std=None
  finite=np.isfinite(uv).all() and np.isfinite(scores).all(); admiss=finite and np.all(scores>=.15) and np.all(border>10)
  if admiss:
   raw,z=tri(uv); fit=least_squares(lambda x:(project(x)-uv).reshape(-1),raw,max_nfev=50); xyz=fit.x; corrected=project(xyz); epi=float(np.linalg.norm((corrected-uv).reshape(-1))); depth=float(np.linalg.norm(xyz));
   try:
    cov=64*np.linalg.inv(fit.jac.T@fit.jac);std=float(np.sqrt(np.linalg.eigvalsh(cov).max()))
   except np.linalg.LinAlgError: std=float('inf')
   if np.min(z)<=0 or depth<.12 or depth>2.5:reason.append('depth_out_of_range')
   if epi>8:reason.append('reprojection_correction_over_8px')
   if std>.30 or not np.isfinite(std):reason.append('depth_uncertainty_over_30cm')
  else:reason.append('low_score_or_border_or_nonfinite')
  temp[name]=dict(uv=uv,scores=scores,border=border,old=old,reason=reason,corrected=corrected,xyz=xyz,cov=cov,epi=epi,depth=depth,std=std)
 for part in ['shoulder','elbow','wrist','hip','knee','ankle']:
  l=temp['left_'+part];r=temp['right_'+part]
  if l['xyz'] is not None and r['xyz'] is not None and np.linalg.norm(l['xyz']-r['xyz'])<.03 and np.max(np.linalg.norm(l['uv']-r['uv'],axis=1))<20:
   l['reason'].append('left_right_duplicate_ambiguous');r['reason'].append('left_right_duplicate_ambiguous')
 for name,q in temp.items():
  ambiguous='left_right_duplicate_ambiguous' in q['reason']; stereo=not q['reason']; v2=(q['scores']>=.3)&(q['border']>40)&np.isfinite(q['uv']).all(axis=1)&(not ambiguous)&('reprojection_correction_over_8px' not in q['reason'])&('depth_out_of_range' not in q['reason']); status='stereo_3d' if stereo else ('2d_only' if np.any(v2) else 'discard'); use=np.ones(2,dtype=bool) if stereo else v2
  row=dict(frame=f,joint=name,status=status,reason='stereo_consistent_candidate' if stereo else ';'.join(q['reason']),old_cam2=int(q['old'][0]),old_cam3=int(q['old'][1]),use_cam2=int(use[0]),use_cam3=int(use[1]),score_cam2=float(q['scores'][0]),score_cam3=float(q['scores'][1]),border_cam2_px=float(q['border'][0]),border_cam3_px=float(q['border'][1]),raw_uv=q['uv'].tolist(),corrected_uv=q['corrected'].tolist() if stereo else None,xyz_cam2_m=q['xyz'].tolist() if stereo else None,cov_cam2_m2=q['cov'].tolist() if stereo else None,correction_norm_px=q['epi'],range_m=q['depth'],largest_sigma_m=q['std'])
  rows.append(row);points[f,name]=row

def sanitize(x):
 if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
 if isinstance(x,list):return [sanitize(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
with (a.out/'decisions.json').open('x') as h:json.dump(sanitize(rows),h,indent=2,allow_nan=False)
print('gating rows',len(rows),flush=True)
