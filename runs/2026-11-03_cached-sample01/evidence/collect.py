import json
from pathlib import Path
import numpy as np
root=Path('outputs/golden_20261103'); evidence=Path('runs/2026-11-03_cached-sample01/evidence'); evidence.mkdir(parents=True,exist_ok=True)
metrics=[]; prov=[]
for i in range(1,4):
 d=root/('run_%02d'%i); v=json.loads((d/'frames/sample_01_cam3/validation.json').read_text())
 keys=['id','passed','matched_hands','body_target_count','body_error_px','hand_error_px','body_rmse_px','hand_rmse_px','body_reload_max_abs_m','fusion_reload_max_abs_m','obj_reload_max_abs_m','minimum_fused_vertex_z_m','pipeline_profile','local_surface_passed']
 metrics.append({'run':i,**{k:v[k] for k in keys if k in v}})
 p=json.loads(next((d/'provenance').glob('*.json')).read_text()); p.pop('expanded_config',None)
 for key in ['pipeline_git','backend_git']:
  state=p.get(key)
  if isinstance(state,dict): state.pop('untracked',None)
 prov.append(p)
 for name,src in [('controller',Path('.local/golden_20261103')/('run_%02d.log'%i)),('frame',d/'sample_01_cam3.log'),('reference',d/'reference.log')]:
  if src.exists():
   text=src.read_text(errors='replace').replace('/vepfs-mlp2/mlp-public/huyaoqing','<WORKSPACE>')
   (evidence/('run_%02d_%s.txt'%(i,name))).write_text(text)
(evidence/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n')
(evidence/'provenance.redacted.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n')
frames=[root/('run_%02d'%i)/'frames/sample_01_cam3' for i in range(1,4)]
comparison={}; files=set(p.relative_to(frames[0]) for p in frames[0].rglob('*.npz'))
for path in sorted(files):
 if not all((f/path).exists() for f in frames):
  comparison[str(path)]={'status':'missing'}; continue
 with np.load(frames[0]/path,allow_pickle=False) as a, np.load(frames[1]/path,allow_pickle=False) as b, np.load(frames[2]/path,allow_pickle=False) as c:
  fields={}
  for key in sorted(set(a.files)&set(b.files)&set(c.files)):
   x,y,z=a[key],b[key],c[key]
   if x.shape!=y.shape or x.shape!=z.shape: fields[key]={'status':'shape_mismatch'}
   elif np.issubdtype(x.dtype,np.number):
    diffs=[float(np.max(np.abs(u.astype(np.float64)-v.astype(np.float64)))) if u.size else 0.0 for u,v in [(x,y),(x,z),(y,z)]]
    fields[key]={'max_abs_pairwise':max(diffs),'shape':list(x.shape)}
   else: fields[key]={'equal':bool(np.array_equal(x,y) and np.array_equal(x,z))}
  comparison[str(path)]={'fields':fields,'keys_equal':set(a.files)==set(b.files)==set(c.files)}
(evidence/'array_comparison.json').write_text(json.dumps(comparison,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print('metric results:',[(m.get('body_rmse_px'),m.get('hand_rmse_px')) for m in metrics])
for name,row in comparison.items():
 vals=[x.get('max_abs_pairwise',0) for x in row.get('fields',{}).values()]
 print(name,max(vals,default=0))
