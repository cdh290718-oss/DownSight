import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=Path(__file__).parent;source=root/'runs/2026-11-16_unrealego-official-noise/evidence';rows=json.loads((source/'results.json').read_text())['records'];verify=json.loads((source/'verification.json').read_text());data={'independent_fits':sum(r['status']=='ok' and not r.get('reused_from') for r in rows),'independent_converged':sum(r['status']=='ok' and not r.get('reused_from') and r['convergence']['diagnostic_pass'] for r in rows),'paired':[],'ranks':[]}
for s in (1,3,5,8):
 d={(r['pair'],r['condition'],r['method']):r for r in rows if r['condition'].startswith(f'gt_s{s}_')}; pairs={(a,b) for a,b,c in d};data['paired'].append({'sigma_px':s,'n':len(pairs),'mixed_pa_less_3d':sum(d[a,b,'mixed']['pa15_mm']<d[a,b,'3d']['pa15_mm'] for a,b in pairs),'2d_pa_less_mixed':sum(d[a,b,'2d']['pa15_mm']<d[a,b,'mixed']['pa15_mm'] for a,b in pairs)}); c=collections.Counter(x['rank'] for x in verify['conditions'] if x['condition'].startswith(f'gt_s{s}_'));data['ranks'].append({'sigma_px':s,'rank_counts':dict(c)})
(p/'evidence/analysis.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
