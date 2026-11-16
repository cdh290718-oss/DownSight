import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[2];run=Path(__file__).resolve().parent;o=root/'outputs/unrealego_official_noise_v2_20261116';out={}
for sigma in [1,3,5,8]:
 cnt=collections.Counter()
 for f in o.glob(f'body*/gt_s{sigma}_seed*/observations.json'):
  for p in json.loads(f.read_text())['points']:
   q=p['tri']
   if sum(p['valid'])<2:cnt['insufficient_2d']+=1
   elif not q['geometry_ok']:cnt['geometry_invalid']+=1
   elif not q['depth_ok']:cnt['depth_unreliable']+=1
   else:cnt['3d_accepted']+=1
 out[str(sigma)]=dict(cnt)
(run/'evidence/screening_reasons.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
