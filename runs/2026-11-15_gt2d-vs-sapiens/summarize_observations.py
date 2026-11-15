import json,statistics,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;src=p/'evidence/observations.json';x=json.loads(src.read_text());points=[q for r in x['pairs'] for q in r['points']];views=[v for q in points for v in q['views']];mean=lambda a:sum(a)/len(a) if a else None
pairs=[]
for r in x['pairs']:
 byview=[]
 for i,side in enumerate(['left','right']):
  vs=[q['views'][i] for q in r['points']];d=[v['distance_px'] for v in vs if v['detected']];byview.append({'camera':side,'gt_in_image':sum(v['official_in_image'] for v in vs),'auto_in_image':sum(v.get('automatic_in_image',False) for v in vs),'score_ge_0_3':sum(v.get('score',0)>=.3 for v in vs),'mean_px':mean(d),'median_px':statistics.median(d) if d else None})
 pairs.append({'pair':r['pair'],'matched_count':r['matched_count'],'views':byview,'matched':[{'name':q['name'],'auto_vs_gt3d_mm':q.get('auto_tri_vs_official_3d_mm'),'gt2d_tri_vs_gt3d_mm':q.get('official_tri_vs_official_3d_mm')} for q in r['points'] if q['matched_accepted']]})
groups=[]
for n in ['shoulder','elbow','wrist','hip','knee','ankle']:
 z=[v['distance_px'] for q in points if q['name'].endswith('_'+n) for v in q['views'] if v['detected']];groups.append({'name':n,'count':len(z),'mean_px':mean(z),'median_px':statistics.median(z) if z else None})
conf=[]
for condition in [True,False]:
 z=[v['distance_px'] for v in views if v['detected'] and (v.get('score',0)>=.3)==condition];conf.append({'score_ge_0_3':condition,'count':len(z),'mean_px':mean(z),'median_px':statistics.median(z) if z else None})
out={'total_view_points':len(views),'gt_in_image':sum(v['official_in_image'] for v in views),'automatic_in_image':sum(v.get('automatic_in_image',False) for v in views),'score_ge_0_3':sum(v.get('score',0)>=.3 for v in views),'visibility':'not assessed; in-frame not equal visible','per_pair':pairs,'per_part':groups,'per_confidence':conf,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'command':'python runs/2026-11-15_gt2d-vs-sapiens/summarize_observations.py','exit_status':0};(p/'evidence/observation_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
