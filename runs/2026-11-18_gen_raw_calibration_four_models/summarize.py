import json,numpy as np
from pathlib import Path
R=Path('runs/2026-11-18_gen_raw_calibration_four_models/evidence');d=json.load(open(R/'3d_results.json'))['records'];p=json.load(open(R/'2d.json'))['records'];out={}
for subset in ('all20','valid19'):
 out[subset]={}
 for m in ('original','rw500','rw5000','rw20000'):
  rows=[x for x in d if x['model']==m and (subset=='all20' or x['pair']!='gen_00000')];ok=[x for x in rows if x['status']=='ok']; e=np.array([x['error_px'] for x in p[m] if subset=='all20' or x['pair']!='gen_00000']);
  out[subset][m]={'pairs':len(rows),'feasible':len(ok),'converged':sum(x['convergence_pass'] for x in ok),'2d_mean_px':float(e.mean()),'2d_count':len(e),'pck20_count':int((e<=20).sum()),'non_pa_mm':float(np.mean([x['absolute_per_joint_mm'] for x in ok])) if ok else None,'pa_mm':float(np.mean([x['pa_per_joint_mm'] for x in ok])) if ok else None,'rmse_mm':float(np.sqrt(np.mean(np.array([x['absolute_per_joint_mm'] for x in ok])**2))) if ok else None}
(R/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
