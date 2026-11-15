import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;source=p/'evidence/results.json';x=json.loads(source.read_text()); rows=[]
for q in x['records']:
 if q['status']!='ok':continue
 rows.append({'pair':q['pair'],'pa15_mm':q['pa15_mean_mm'],'absolute15_mm':q['absolute15_mean_mm'],'prior_direct_gt3d_pa15_mm':q['prior_gt3d_pa15_mm'],'historical_pa15_mm':sum(v['historical_pa15_mm'] for v in q['views'])/2,'2d_mean_px':[v['2d_mean_px'] for v in q['views']],'shape_rank':q['shape_rank'],'beta0':q['betas_final'][0],'convergence_pass':q['convergence']['diagnostic_pass']})
assert len(rows)==4
keys=['pa15_mm','absolute15_mm','prior_direct_gt3d_pa15_mm','historical_pa15_mm'];y={'rows':rows,'averages':{k:sum(r[k] for r in rows)/len(rows) for k in keys},'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'command':'python runs/2026-11-15_unrealego-gt2d/summarize.py','exit_status':0};(p/'evidence/summary.json').write_text(json.dumps(y,indent=2,allow_nan=False)+'\n');print(json.dumps(y,indent=2))
