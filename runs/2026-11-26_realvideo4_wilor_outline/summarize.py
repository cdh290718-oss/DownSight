"""Pool landmark squared distances and normalize visibility-weighted contour distances."""
import math
import json
import statistics as stats
from pathlib import Path

RUN=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
rows={f.stem:read(f) for f in sorted((RUN/'evidence').glob('clip*.json'))}
assert len(rows)==4
guide=read(RUN.parents[1]/'outputs/realvideo4_palm_20261126/guided.json')['selected']
summary={}
for mode in ('start','control','wilor'):
    cases=[d['results'][mode] for d in rows.values()]
    hands=[h for case in cases for h in case['hands'].values()]
    weight=sum(v['weight'] for d in rows.values() for v in d['visibility'].values())
    summary[mode]={key:math.sqrt(stats.mean(h[key]**2 for h in hands)) for key in ('wrist_px','root_px','tips_px','thumbtip_px','pinkytip_px')}
    summary[mode].update(segment_hand_edge_px=stats.mean(v for h in hands for v in h['segment_edge_px'] if v is not None),
                         wilor_hand_edge_px=sum(c['wilor_weighted_edge_px']*4 for c in cases)/weight,
                         segment_edge_px=stats.mean(c['segment_edge_px'] for c in cases),
                         max_body_joint_shift_mm=max(c['body_joint_max_mm'] for c in cases),
                         normal_vs_stereo_deg=stats.mean(h['normal_vs_stereo'] for h in hands))
wrists={}
for key,d in rows.items():
    for i,eye in enumerate(('left','right')):
        for side,h in d['results']['control']['hands'].items():
            p=guide[key+'_'+eye][side[0].upper()]['uv'][0]
            q=h['target'][i][0]
            wrists[key+'_'+eye+'_'+side]=math.hypot(p[0]-q[0],p[1]-q[1])
worse=sum(d['results']['wilor']['hands'][s]['tips_px']>d['results']['control']['hands'][s]['tips_px'] for d in rows.values() for s in ('left','right'))
(RUN/'evidence/summary.json').write_text(json.dumps(dict(stages=summary,wilor_vs_detected_wrist_px=wrists,hands_with_increased_tip_rmse=worse,exit_status=0),indent=2)+'\n')
