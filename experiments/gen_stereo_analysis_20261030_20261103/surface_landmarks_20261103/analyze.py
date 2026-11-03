"""Candidate observations only; no fitting or non-wrist DFM calibration."""
import argparse, ast, json, csv, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cal',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
cal=json.loads(a.cal.read_text()); ds=[np.array(cal['cam2_ds']),np.array(cal['cam3_ds'])];R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']); tree=ast.parse(a.geometry.read_text()); module=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('unproj','proj','project','tri')],type_ignores=[]);exec(compile(module,str(a.geometry),'exec'))
d=np.load(a.data/'det_sapiens2.npz');ref=np.load(a.data/'dfm_reference.npz');names=d['names'].tolist(); frames=list(range(0,5131,270)); idx={(int(c),int(f)):i for i,(c,f) in enumerate(zip(d['cams'],d['frames']))}; ridx={int(f):i for i,f in enumerate(ref['frames'])}
core=['nose']+[s+'_'+p for p in ('shoulder','elbow','wrist','hip','knee','ankle') for s in ('left','right')];surfaces=[s+'_'+p for p in ('olecranon','cubital_fossa','acromion','big_toe','small_toe','heel') for s in ('left','right')]; intermediate=[n for n in names[21:63] if n not in ('left_wrist','right_wrist') and not n.endswith('4')]; targets=core+surfaces+intermediate
# Robust noise scale is reference disagreement, not ground-truth detector noise.
residual=[]
for f in frames:
 for c in (2,3):
  i=idx[c,f];k=ridx[f]
  for side,j in [('left',20),('right',21)]:
   q=names.index(side+'_wrist');uv=d['xy'][i,q]; score=float(d['score'][i,q]);gt=ref['cam%d_body_uv'%c][k,j];valid=bool(ref['cam%d_body_valid'%c][k,j]);
   if valid and np.isfinite(uv).all() and np.isfinite(gt).all() and score>=.3 and 0<=uv[0]<1600 and 0<=uv[1]<1300 and 0<=gt[0]<1600 and 0<=gt[1]<1300:
    residual.append(dict(frame=f,camera=c,joint=side+'_wrist',dx=float(uv[0]-gt[0]),dy=float(uv[1]-gt[1])))
rv=np.array([[x['dx'],x['dy']] for x in residual]); med=np.median(rv,axis=0);sigma=float(1.4826*np.median(np.abs(rv-med))); assert sigma>0
rows=[]; lookup={}
for f in frames:
 for name in targets:
  j=names.index(name);xy=np.array([d['xy'][idx[c,f],j] for c in (2,3)],float);sc=np.array([d['score'][idx[c,f],j] for c in (2,3)],float); good=np.isfinite(xy).all(axis=1)&np.isfinite(sc)&(sc>=.3)&(xy[:,0]>=0)&(xy[:,0]<1600)&(xy[:,1]>=0)&(xy[:,1]<1300);xyz=None;cov=None;corr=None;depth=None;standard=None; reason=[]
  if good.all():
   try:
    raw,z=tri(xy);fit=least_squares(lambda x:(project(x)-xy).reshape(-1),raw,max_nfev=100);xyz=fit.x;corr=float(np.linalg.norm(project(xyz)-xy));depth=float(np.linalg.norm(xyz));cov=sigma**2*np.linalg.pinv(fit.jac.T@fit.jac);standard=float(np.sqrt(max(0,np.linalg.eigvalsh(cov).max())));
    if min(z)<=0 or not .12<=depth<=2.5:reason.append('depth')
    if corr>8:reason.append('reprojection')
    if standard>.30:reason.append('uncertainty')
   except (ValueError,np.linalg.LinAlgError,FloatingPointError):reason.append('geometry_failure')
  else:reason.append('not_two_eligible_views')
  stereo=not reason
  # Head fitting excluded by brief. A body joint center is not a surface observation.
  if name=='nose': use='out_of_scope'
  elif name.endswith('_wrist') and stereo:use='3d_candidate'
  elif name in core and any(good) and 'reprojection' not in reason and 'depth' not in reason:use='2d_candidate'
  elif name in surfaces and stereo:use='surface_3d_candidate'
  elif name in surfaces and any(good) and 'reprojection' not in reason and 'depth' not in reason:use='surface_2d_candidate'
  elif name in intermediate and stereo:use='hand_3d_candidate'
  elif name in intermediate and any(good):use='hand_2d_candidate'
  else:use='discard'
  row=dict(frame=f,name=name,index=j,score=sc.tolist(),xy=xy.tolist(),eligible=good.tolist(),stereo=stereo,reasons=reason,correction_l2_px=corr,range_m=depth,sigma_max_m=standard,xyz_cam2_m=xyz.tolist() if stereo else None,cov_cam2_m2=cov.tolist() if stereo else None,recommended=use)
  rows.append(row);lookup[f,name]=row
# Treat a cross-side collision as an unresolved identity, not two valid constraints.
for f in frames:
 for part in ('shoulder','elbow','wrist','hip','knee','ankle','olecranon','cubital_fossa','acromion'):
  l=lookup[f,'left_'+part];r=lookup[f,'right_'+part]
  if l['stereo'] and r['stereo'] and np.linalg.norm(np.array(l['xyz_cam2_m'])-np.array(r['xyz_cam2_m']))<.03 and np.max(np.linalg.norm(np.array(l['xy'])-np.array(r['xy']),axis=1))<20:
   for q in (l,r):q['stereo']=False;q['reasons'].append('left_right_duplicate_ambiguous');q['xyz_cam2_m']=None;q['cov_cam2_m2']=None;q['recommended']='discard'
# Paired hypothesis was fixed before looking at DFM elbows: midpoint of two markers.
tests=[]
for f in frames:
 for side,di in [('left',18),('right',19)]:
  elbow=lookup[f,side+'_elbow'];ole=lookup[f,side+'_olecranon'];cub=lookup[f,side+'_cubital_fossa'];k=ridx[f]
  if elbow['stereo'] and ole['stereo'] and cub['stereo'] and bool(ref['cam2_body_valid'][k,di]):
   gt=ref['cam2_body_xyz'][k,di];pred=(np.array(ole['xyz_cam2_m'])+np.array(cub['xyz_cam2_m']))/2; direct=np.array(elbow['xyz_cam2_m']);tests.append(dict(frame=f,side=side,midpoint_error_cm=float(np.linalg.norm(pred-gt)*100),elbow_error_cm=float(np.linalg.norm(direct-gt)*100),marker_separation_cm=float(np.linalg.norm(np.array(ole['xyz_cam2_m'])-np.array(cub['xyz_cam2_m']))*100)))
# Endpoint support is a coarse proxy, never proof of visibility or observability.
coverage=[]
for f in frames:
 for side in ('left','right'):
  def any_source(parts,kind):
   result=[]
   for n in parts:
    r=lookup[f,side+'_'+n]
    if kind=='old' and n in ('shoulder','elbow','wrist','hip','knee','ankle') and any(r['eligible']):result.append(n)
    if kind=='candidate' and (r['stereo'] if n in ('olecranon','cubital_fossa','acromion','big_toe','small_toe','heel','wrist') else r['recommended']=='2d_candidate'):result.append(n)
   return result
  for segment,prox,dist in [('upper_arm',['shoulder','acromion'],['elbow','olecranon','cubital_fossa']),('forearm',['elbow','olecranon','cubital_fossa'],['wrist']),('thigh',['hip'],['knee']),('shin',['knee'],['ankle','big_toe','small_toe','heel']),('torso_pelvis',['shoulder','acromion'],['hip'])]:
   for kind in ('old','candidate'):
    x=any_source(prox,kind);y=any_source(dist,kind);coverage.append(dict(frame=f,side=side,segment=segment,method=kind,proximal=';'.join(x),distal=';'.join(y),two_endpoint_proxy=bool(x and y)))
# CSV statistics separate 2-view eligibility and conditional stereo, not visible ground truth.
stats=[]
for n in targets:
 rr=[r for r in rows if r['name']==n];n0=sum(r['eligible'][0] for r in rr);n1=sum(r['eligible'][1] for r in rr);both=sum(all(r['eligible']) for r in rr);passed=sum(r['stereo'] for r in rr);stats.append(dict(name=n,index=names.index(n),eligible_cam2=n0,eligible_cam3=n1,both_eligible=both,stereo_pass=passed,conditional_pass_fraction=passed/both if both else None))
with (a.out/'decisions.json').open('w') as h:json.dump(rows,h,indent=2,allow_nan=False)
for file,data in [('stats.csv',stats),('coverage.csv',coverage),('elbow_test.csv',tests),('wrist_residuals.csv',residual)]:
 with (a.out/file).open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=(list(data[0]) if data else ['no_paired_samples']),lineterminator='\n');w.writeheader();w.writerows(data)
summary=dict(frames=frames,wrist_sample_count=len(residual),wrist_median_xy_px=med.tolist(),wrist_sigma_mad_px=sigma,wrist_absolute_error_quantiles_px=np.percentile(np.linalg.norm(rv,axis=1),[50,90,95]).tolist(),stereo_cut_px=8,elbow_pairs=len(tests),elbow_tests=tests,old_coverage=sum(x['two_endpoint_proxy'] for x in coverage if x['method']=='old'),candidate_coverage=sum(x['two_endpoint_proxy'] for x in coverage if x['method']=='candidate'),input_sha256={n:hashlib.sha256((a.data/n).read_bytes()).hexdigest() for n in ('det_sapiens2.npz','dfm_reference.npz')})
with (a.out/'summary.json').open('w') as h:json.dump(summary,h,indent=2,allow_nan=False)
print(json.dumps(summary,indent=2))
