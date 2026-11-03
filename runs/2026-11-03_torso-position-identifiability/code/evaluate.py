"""Post-fit evaluation. DFM is read here only; wrist surface uses stereo proxy, not DFM."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ref=np.load(a.reference,allow_pickle=False);obs=np.load(a.observations,allow_pickle=False);cal=json.loads(a.calibration.read_text());lookup={int(f):i for i,f in enumerate(ref['frames'])}
def project(points,ds):
 fx,fy,cx,cy,xi,alpha=ds;x,y,z=points.T;d1=np.sqrt(x*x+y*y+z*z+1e-12);z1=xi*d1+z;d2=np.sqrt(x*x+y*y+z1*z1+1e-12);den=np.maximum(alpha*d2+(1-alpha)*z1,1e-6);return np.column_stack([fx*x/den+cx,fy*y/den+cy])
def stats(values):
 x=np.asarray(values,dtype=float);return {'count':len(x),'median_cm':float(np.median(x)*100) if len(x) else None,'per_frame_cm':(x*100).tolist()}
out={'method':'DFM is post-fit only. Wrist joint uses DFM joint; wrist surface uses stereo triangulation of Sapiens wrist pixel as an unvalidated visible-surface proxy, not DFM surface truth.','splits':{'train':obs['frames'][:10].tolist(),'test':obs['frames'][10:].tolist()},'conditions':{}}
selection=json.loads((a.results/'long_selection.json').read_text())
for condition in ('baseline','dense','baseline_long','dense_long'):
 rows=[]; centers=[]
 seeds=selection[condition[:-5]]['seeds'] if condition.endswith('_long') else range(7)
 for seed in seeds:
  folder=a.results/f'{condition}_{seed}';s=json.loads((folder/'settings.json').read_text());d=np.load(folder/'test.npz',allow_pickle=False);tr=np.load(folder/'train.npz',allow_pickle=False);ids=[lookup[int(f)] for f in d['frame_ids']];gt=ref['cam2_body_xyz'][ids];ok=ref['cam2_body_valid'][ids];center=d['shared_head_center_cam2_m'];centers.append(center);head=d['head_joint_cam2']
  headmask=ok[:,15];h=np.linalg.norm(head-gt[:,15],axis=1);hc=np.linalg.norm(center[None,:]-gt[:,15],axis=1)
  wrists={}
  for side,si,gi in [('left',12,20),('right',13,21)]:
   oi=obs['names'].tolist().index(side+'_wrist');truth=gt[:,gi];direction=truth/np.maximum(np.linalg.norm(truth,axis=1,keepdims=True),1e-8);joint=np.abs(((d['joints_cam2'][:,si]-truth)*direction).sum(-1));tri=np.abs(((obs['tri'][10:,oi]-truth)*direction).sum(-1));valid=ok[:,gi]&obs['tri_valid'][10:,oi]&np.isfinite(joint)&np.isfinite(tri)
   errors=[];frames=[];pixel_errors=[]
   for i in range(len(ids)):
    if not obs['tri_valid'][10+i,oi] or not obs['valid'][10+i,0,oi]:continue
    v=d['vertices_cam2'][i];j=d['joints_cam2'][i,si];cand=v[(np.linalg.norm(v-j,axis=1)<.08)&(v[:,2]>.01)]
    if not len(cand):continue
    uv=project(cand,cal['cam2_ds']);dist=np.linalg.norm(uv-obs['xy'][10+i,0,oi],axis=1);idx=int(np.argmin(dist))
    if dist[idx]>20:continue
    reference=obs['tri'][10+i,oi];unit=reference/np.linalg.norm(reference);errors.append(abs(float((cand[idx]-reference)@unit)));frames.append(int(d['frame_ids'][i]));pixel_errors.append(float(dist[idx]))
   wrists[side]={'joint_depth_vs_dfm':stats(joint[valid]),'triangulated_wrist_vs_dfm':stats(tri[valid]),'surface_depth_vs_stereo_proxy':stats(errors),'surface_valid_frames':frames,'surface_pixel_distance_px':pixel_errors}
  rows.append({'seed':seed,'initial_center_cam2_m':s['initial_center_cam2_m'],'shared_center_cam2_m':center.tolist(),'shared_vs_dfm_head_cm':stats(hc[headmask]),'fitted_head_vs_dfm_cm':stats(h[headmask]),'wrists':wrists,'train_loss':float(tr['final_loss']),'test_loss':float(d['final_loss'])})
 centers=np.array(centers);out['conditions'][condition]={'max_pairwise_center_cm':float(max(np.linalg.norm(x-y) for x in centers for y in centers)*100),'coordinate_range_cm':(np.ptp(centers,axis=0)*100).tolist(),'runs':rows}
with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
