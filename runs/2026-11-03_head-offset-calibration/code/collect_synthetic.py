"""Collect all expected SMPL-X cases, including failures, without source imagery."""
import argparse,json,re
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();conditions=[]
for axes in (1,2):
 for deg in (0,10,20,40,60):
  cases=[]
  for trial in (0,1):
   name=f'axes{axes}_deg{deg}_trial{trial}';path=a.results/(name+'.json');log=a.results/(name+'.txt')
   if path.exists():cases.append({'status':'completed','result':json.loads(path.read_text())})
   else:
    excerpt=log.read_text() if log.exists() else 'missing case log';excerpt=re.sub(r'/(?:vepfs-mlp2|root|home)/[^ :]*','<REDACTED_PATH>',excerpt)
    cases.append({'status':'failed','log':excerpt})
  ok=[c['result'] for c in cases if c['status']=='completed'];conditions.append({'axes':axes,'per_axis_amplitude_deg':deg,'completed':len(ok),'spread_cm': [r['max_pairwise_cm'] for r in ok],'worst_start_error_cm':[max(r['error_cm']) for r in ok],'max_actual_neck_rotation_deg':[r['max_neck_rotation_deg'] for r in ok],'cases':cases})
long=[]
for axes,deg in ((2,0),(1,40),(1,60),(2,40),(2,60)):
 path=a.results/f'long_axes{axes}_deg{deg}_trial0.json';log=a.results/f'long_axes{axes}_deg{deg}_trial0.txt'
 if path.exists():long.append({'status':'completed','result':json.loads(path.read_text())})
 else:long.append({'status':'not_completed','axes':axes,'amplitude_deg':deg,'log':re.sub(r'/(?:vepfs-mlp2|root|home)/[^ :]*','<REDACTED_PATH>',log.read_text()) if log.exists() else 'not run'})
with a.output.open('x') as f:json.dump({'note':'Each synthetic trial uses 12 frames, 7 initial offsets, the same known true shape and exact correspondences; two trials are not a statistical guarantee.','conditions':conditions,'long_checks':long},f,indent=2,allow_nan=False);f.write('\n')
