"""Aggregate fixed-denominator groups without hiding missing views."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();data=json.loads(a.metrics.read_text());summary={}
for subset in ['all','new','old']:
 rows=[r for r in data['frames'] if subset=='all' or r['pair'].startswith(subset)];methods={}
 for method in ['original','edge40','no_margin']:
  q={};valid=[r['methods'][method] for r in rows if method in r['methods'] and not r['methods'][method].get('failed')]
  for key in ['body_all','body_retained','body_shoulders']:
   values=[v[key]['rmse'] for v in valid if v[key]['rmse'] is not None];q[key]=dict(mean_view_rmse_px=float(np.mean(values)) if values else None,views=len(values))
  for key in ['all','retained']:
   values=[h[key]['rmse'] for v in valid for h in v['hands'].values() if h and h[key]['rmse'] is not None];q['hand_'+key]=dict(mean_hand_view_rmse_px=float(np.mean(values)) if values else None,hand_views=len(values))
  for key in ['triangulated_3d_m','triangulated_depth_m','triangulated_transverse_m']:
   values=[h[key]['rmse'] for v in valid for h in v['hands'].values() if h and h[key]['rmse'] is not None];q[key]=dict(mean_hand_view_rmse_m=float(np.mean(values)) if values else None,hand_views=len(values))
  values=[h['triangulated_wrist_m'] for v in valid for h in v['hands'].values() if h and h['triangulated_wrist_m'] is not None];q['triangulated_wrist_m']=dict(mean_distance_m=float(np.mean(values)) if values else None,hand_views=len(values))
  q['head_visible_vertices_total']=sum(v['head_visible'] for v in valid);q['local_crossings_total']=sum(h['proper_crossings'] for v in valid for h in v['hands'].values() if h);q['views']=len(valid)
  methods[method]=q
 summary[subset]=methods
out=dict(aggregation='Arithmetic mean of per-view body RMSE or per-side/per-view hand RMSE, fixed samples; automatic reference only',subsets=summary)
a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
