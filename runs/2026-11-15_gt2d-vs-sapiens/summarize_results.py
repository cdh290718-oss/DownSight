import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;src=p/'evidence/results.json';x=json.loads(src.read_text());old=json.loads((p.parents[1]/'runs/2026-11-15_unrealego-gt2d/evidence/summary.json').read_text());prep=json.loads((p/'evidence/preparation.json').read_text());rows=[]
for i in prep['pairs']:
 name=i['pair']; mapping={'left_shoulder':'L_Shoulder','right_shoulder':'R_Shoulder','left_elbow':'L_Elbow','right_elbow':'R_Elbow','left_wrist':'L_Wrist','right_wrist':'R_Wrist'};row={'pair':name,'matched_count':i['matched'],'matched_names':i['names'],'official_full_15_pa_mm':next(r['pa15_mm'] for r in old['rows'] if r['pair']==name)}
 for mode in ['auto','official']:
  q=next((r for r in x['records'] if r.get('pair')==name and r.get('mode')==mode),None)
  if q and q['status']=='ok':row[mode]={'pa15_mm':q['pa15_mean_mm'],'absolute15_mm':q['absolute15_mean_mm'],'convergence_pass':q['convergence']['diagnostic_pass'],'shape_rank':q['shape_rank'],'matched_absolute_per_joint_mm':{n:q['absolute15_per_joint_mm'][mapping[n]] for n in i['names']},'matched_absolute_mean_mm':sum(q['absolute15_per_joint_mm'][mapping[n]] for n in i['names'])/len(i['names'])}
  else:row[mode]=None
 rows.append(row)
out={'note':'same-joint paired conditions use identical covariance and accepted targets; full15 reference has different target count and weighting; output 3D GT excluded from fitting except Head and rig calibration','rows':rows,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'command':'python runs/2026-11-15_gt2d-vs-sapiens/summarize_results.py','exit_status':0};(p/'evidence/summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
