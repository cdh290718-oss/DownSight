"""Reuse fixed diagnostic masks and anchors; adapt new predictions to camera pixels."""
import json,hashlib
from pathlib import Path
run=Path('runs/2026-11-17_newdetector_d10');out=Path('outputs/newdetector_d10_20261117');prev=json.loads(Path('outputs/2d_intervention_20261117_retry1/prepared.json').read_text())['items'];pred={r['id']:r for r in json.loads((out/'predictions.json').read_text())};ids={'R_Shoulder':6,'R_Elbow':8,'R_Wrist':41,'L_Shoulder':5,'L_Elbow':7,'L_Wrist':62,'R_Hip':10,'R_Knee':12,'R_Ankle':14,'L_Hip':9,'L_Knee':11,'L_Ankle':13};scale=720/872;offset=[279.91284403669727,-0.08715596330275227];rows=[]
for old in prev:
 if old['condition']!='baseline':continue
 for p in old['points']:
  for v,side in enumerate(['left','right']):
   x=pred[old['pair']+'_'+side]['xy'][ids[p['name']]];p['xy'][v]=[x[0]*scale+offset[0],x[1]*scale+offset[1]];p['scores'][v]=pred[old['pair']+'_'+side]['score'][ids[p['name']]]
  p['accepted']=False;p.pop('xyz',None);p.pop('covariance',None)
 rows.append(old)
assert len(rows)==10
(out/'prepared.json').open('x').write(json.dumps({'items':rows,'protocol_sha256':hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()},indent=2))
