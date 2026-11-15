import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]; run=Path(__file__).resolve().parent
source=root.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1/body_metrics.json';p=run/'evidence/results.json';data=json.loads(p.read_text());history=json.loads(source.read_text());rows=[]
for r in data['records']:
 if r['status']!='ok':continue
 pair=r['pair'];views=[next(v for v in history['per_image'] if v['id']==pair+'_'+s) for s in ['left','right']]; vals=[v['pa_mpjpe_mm'] for v in views];rows.append({'pair':pair,'D_pa15_mm':r['pa15_mean_mm'],'D_absolute15_mm':r['absolute15_mean_mm'],'D_head_absolute_mm':r['head_absolute_error_mm'],'D_absolute16_mm':(15*r['absolute15_mean_mm']+r['head_absolute_error_mm'])/16,'historical_pa15_views_mm':vals,'historical_pa15_pair_mm':sum(vals)/len(vals),'convergence_pass':r['convergence']['diagnostic_pass']})
assert len(rows)==4
avg=lambda k:sum(x[k] for x in rows)/len(rows)
q={'scope':'4 predefined samples; D used evaluated GT points during fitting, historical did not','history_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'D_results_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':rows,'averages':{k:avg(k) for k in ['D_pa15_mm','D_absolute15_mm','D_head_absolute_mm','D_absolute16_mm','historical_pa15_pair_mm']},'command':'python runs/2026-11-15_unrealego-gt16-four/compare.py','exit_status':0};(run/'evidence/comparison.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps(q,indent=2))
