"""Apply published gating decisions without DFM and recalculate stereo geometry in the fit calibration."""
import argparse,json,os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--published',type=Path,required=True);p.add_argument('--recomputed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
src=np.load(a.input/'observations.npz',allow_pickle=False); cal=json.loads((a.input/'camera_calibration.json').read_text());R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);ds=[cal['cam2_ds'],cal['cam3_ds']];names=src['names'].tolist();frames=src['frames'];published=json.loads(a.published.read_text());recomputed=json.loads(a.recomputed.read_text());lookups=[{(int(x['frame']),x['joint']):x for x in rows} for rows in (published,recomputed)]
def unproj(uv,d):
 fx,fy,cx,cy,xi,al=d;x=(uv[0]-cx)/fx;y=(uv[1]-cy)/fy;r=x*x+y*y;z=(1-al*al*r)/(al*np.sqrt(max(0,1-(2*al-1)*r))+1-al);k=(z*xi+np.sqrt(max(0,z*z+(1-xi*xi)*r)))/(z*z+r);v=np.array([k*x,k*y,k*z-xi]);return v/np.linalg.norm(v)
def proj(x,d):
 fx,fy,cx,cy,xi,al=d;n=np.linalg.norm(x);z=xi*n+x[2];den=al*np.sqrt(x[0]**2+x[1]**2+z*z)+(1-al)*z;return np.array([fx*x[0]/den+cx,fy*x[1]/den+cy])
def project(x):return np.array([proj(x,ds[0]),proj(R.T@(x-t),ds[1])])
def stereo(uv):
 v=unproj(uv[0],ds[0]);w=R@unproj(uv[1],ds[1]);z=np.linalg.lstsq(np.stack([v,-w],axis=1),t,rcond=None)[0];init=(v*z[0]+t+w*z[1])/2;f=least_squares(lambda x:(project(x)-uv).reshape(-1),init,max_nfev=50)
 try:cov=64*np.linalg.inv(f.jac.T@f.jac);vals,V=np.linalg.eigh(cov);valid=(np.min(z)>0 and np.isfinite(vals).all() and np.isfinite(f.x).all() and vals.min()>0);vals=np.clip(vals,.008**2,.5**2);W=(V*(1/np.sqrt(vals)))@V.T
 except np.linalg.LinAlgError:valid=False;W=np.zeros((3,3))
 return f.x,project(f.x),W,valid
summary={};a.output.mkdir(exist_ok=False,parents=True)
for mode in ('baseline','published_gate','recomputed_gate','wrist_only','published_gate_wrist_only'):
 arrays={k:src[k].copy() for k in src.files}; decisions=[];which=lookups[0 if mode.startswith('published') else 1]; gated=mode in ('published_gate','recomputed_gate','published_gate_wrist_only'); wristonly=mode in ('wrist_only','published_gate_wrist_only')
 for fi,frame in enumerate(frames):
  for j,name in enumerate(names):
   if gated:
    row=which[(int(frame),name)]; arrays['valid'][fi,:,j]=[bool(row['use_cam2']),bool(row['use_cam3'])]; arrays['tri_valid'][fi,j]=False; arrays['tri'][fi,j]=0;arrays['whiten'][fi,j]=0
    if row['status']=='stereo_3d':
     pt,uv,W,ok=stereo(np.asarray(row['raw_uv'],float))
     if ok:arrays['xy'][fi,:,j]=uv;arrays['tri'][fi,j]=pt;arrays['whiten'][fi,j]=W;arrays['tri_valid'][fi,j]=True
     else:arrays['valid'][fi,:,j]=False
    decisions.append({'frame':int(frame),'joint':name,'published_or_recomputed_status':row['status'],'reason':row['reason'],'used_cam2':bool(arrays['valid'][fi,0,j]),'used_cam3':bool(arrays['valid'][fi,1,j]),'used_stereo':bool(arrays['tri_valid'][fi,j])})
   if wristonly and 'wrist' not in name and 'elbow' not in name:
    arrays['valid'][fi,:,j]=False;arrays['tri_valid'][fi,j]=False
 dest=a.output/mode;dest.mkdir();(dest/'camera_calibration.json').write_text(json.dumps(cal,indent=2)+'\n');os.symlink((a.input/'initial').resolve(),dest/'initial',target_is_directory=True)
 with (dest/'observations.npz').open('xb') as f:np.savez_compressed(f,**arrays)
 summary[mode]={'test_cam2_by_name':{n:int(arrays['valid'][10:,0,j].sum()) for j,n in enumerate(names)},'test_stereo_by_name':{n:int(arrays['tri_valid'][10:,j].sum()) for j,n in enumerate(names)},'decisions':decisions}
with (a.output/'summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
