"""Check routing and geometry record invariants on prepared inputs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'outputs/unrealego_official_noise_v2_20261116';summary=json.loads((Path(__file__).parent/'evidence/preparation.json').read_text()); count=0;fall=0;single=0
for item in summary['conditions']:
 x=json.loads((out/item['pair']/item['condition']/'observations.json').read_text())
 for p in x['points']:
  a=p['accepted'];valid=sum(p['valid']);tri=p['tri'];assert a==tri['depth_ok'];assert not a or valid==2 and tri['geometry_ok'] and tri['depth_std_m']<=.5
  for method in ('3d','mixed','2d'):
   use3=bool(a and method!='2d');use2=bool(method!='3d' and valid and not use3);assert not(use3 and use2)
   if method=='mixed' and valid and not a:assert use2
  fall+=int(not a and valid>0);single+=int(valid==1);count+=1
r={'exit_status':0,'points_checked':count,'fallback_points':fall,'single_view_points':single,'check':'routing and preparation invariants, not numerical fit accuracy'}
(Path(__file__).parent/'evidence/routing_test.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
