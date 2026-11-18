import json,hashlib,copy
from pathlib import Path
r=Path('runs/2026-11-18_gen_original_comparison');out=Path('outputs/gen_original_comparison_20261118');previous=json.loads(Path('outputs/gen_three_models_20261118/prepared.json').read_text())['items'];pred={x['id']:x for x in json.loads(Path('outputs/gen_sapiens_compare_20261117/original.json').read_text())};rows=[]
for orig in previous:
 if orig['condition']!='rw500':continue
 row=copy.deepcopy(orig);row['condition']='original'
 for point in row['points']:
  mapped=('right_' if point['name'].startswith('R_') else 'left_')+point['part']
  for k,cam in enumerate(['cam3','cam4']):
   obj=pred[row['pair']+'_'+cam];p=next(x for x in obj['points'] if x['name']==mapped);assert bool(point['valid'][k])==p['reference_in_image'];point['xy'][k]=p['prediction'];point['scores'][k]=p['score']
 rows.append(row)
assert len(rows)==20
(out/'prepared.json').open('x').write(json.dumps({'items':rows,'protocol_sha256':hashlib.sha256((r/'protocol.json').read_bytes()).hexdigest()},indent=2))
