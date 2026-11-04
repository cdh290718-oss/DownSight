"""Supplementary earlier-frame check; calibration uses these frames' DFM heads, not their wrists."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ref=np.load(a.reference,allow_pickle=False);obs=np.load(a.inputs/'baseline/observations.npz',allow_pickle=False);frames=obs['frames'][:10];lookup={int(f):i for i,f in enumerate(ref['frames'])};ix=[lookup[int(f)] for f in frames];gt=ref['cam2_body_xyz'][ix];valid=ref['cam2_body_valid'][ix];cal=json.loads((a.inputs/'baseline/camera_calibration.json').read_text());R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);names=obs['names'].tolist()
def project(x,ds):
 fx,fy,cx,cy,xi,alpha=ds;x,y,z=x.T;d=np.sqrt(x*x+y*y+z*z);z1=xi*d+z;den=alpha*np.sqrt(x*x+y*y+z1*z1)+(1-alpha)*z1;return np.stack((fx*x/den+cx,fy*y/den+cy),axis=-1)
results={};fits={}
for mode in ('baseline','published_gate_cross_hand'):
 for step in (500,1500):
  key=f'{mode}_{step}';d=np.load(a.results/f'first10_{key}'/'test.npz',allow_pickle=False);assert np.array_equal(d['frame_ids'],frames);fits[key]=d;side_results={}
  for side,si,gi in [('left',12,20),('right',13,21)]:
   err=np.linalg.norm(d['joints_cam2'][:,si]-gt[:,gi],axis=1)*100;mask=valid[:,gi]&np.isfinite(err);x=err[mask];side_results[side]={'count':len(x),'median_cm':float(np.median(x)),'p90_cm':float(np.percentile(x,90)),'valid_frames':frames[mask].tolist(),'per_frame_cm':x.tolist()}
  results[key]=side_results
rows=[]
for i,f in enumerate(frames):
 row={'frame':int(f),'wrists':{}}
 for side,si,gi in [('left',12,20),('right',13,21)]:
  ok=bool(valid[i,gi]);truth=gt[i,gi];j=names.index(side+'_wrist');tri=bool(obs['tri_valid'][i,j]);uvgt=[project(truth[None,:],cal['cam2_ds'])[0].tolist(),project(((truth-t)@R)[None,:],cal['cam3_ds'])[0].tolist()] if ok else None
  info={'reference_valid':ok,'observation_stereo_valid':tri,'observation_vs_dfm_cm':float(np.linalg.norm(obs['tri'][i,j]-truth)*100) if ok and tri else None,'raw_uv':obs['xy'][i,:,j].tolist(),'reference_projected_uv':uvgt,'fits':{}}
  for key,d in fits.items():
   xyz=d['joints_cam2'][i,si];xyz3=(xyz-t)@R;info['fits'][key]={'vs_dfm_cm':float(np.linalg.norm(xyz-truth)*100) if ok else None,'projected_uv':[project(xyz[None,:],cal['cam2_ds'])[0].tolist(),project(xyz3[None,:],cal['cam3_ds'])[0].tolist()]}
  info['bad_frame']=bool(ok and max(z['vs_dfm_cm'] for z in info['fits'].values())>=5);row['wrists'][side]=info
 rows.append(row)
with a.output.open('x') as f:json.dump({'note':'Earlier frames used to derive DFM-head simulated calibration; wrist labels were not used in fitting, but frames and participant were used in previous development. Not an independent dataset.','frames':frames.tolist(),'variants':results,'per_frame':rows},f,indent=2,allow_nan=False);f.write('\n')
