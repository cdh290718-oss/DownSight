"""Choose a reproducible diagnostic set from already evaluated outputs."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--hand',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--groups',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=json.loads(a.metrics.read_text());h=json.loads(a.hand.read_text());v=json.loads(a.validation.read_text());g=json.loads(a.groups.read_text());rows=m['per_frame']['two_stage'];hands={mode:{int(x['frame']):x['hands'] for x in h[mode]} for mode in ('v0','two_stage')};noobs={int(x['frame']) for x in v['unobserved_hand_joint_drift']['left']['per_frame']};by={int(x['frame']):x for x in rows};ranked=sorted([x for x in rows if all(x['sides'][s]['wrist_error_cm'] is not None for s in ('left','right'))],key=lambda x:(max(x['sides'][s]['wrist_error_cm'] for s in ('left','right')),x['frame']));selected={}
def add(frame,reason):
 frame=int(frame)
 if frame not in selected:selected[frame]=[]
 selected[frame].append(reason)
# Three mandatory situations: fixed frame 00270, first 21-point near-hands frame, and median-error left-no-observation frame.
add(270,'右腕在 cam2 画外（指定帧）');near=sorted(g['any_hand_point_lt_5cm_exploratory_frame_ids']);add(near[0],'双手最近 DFM 参考点距 <5 cm（探索性分组）')
ordered_noobs=[x for x in ranked if int(x['frame']) in noobs and x['frame'] not in selected];add(ordered_noobs[len(ordered_noobs)//2]['frame'],'左手无可用 cam2 二维或双目三维手点；按双腕最大误差取中位')
# For each bucket, take nearest available deterministic percentile ranks, excluding already selected frames.
N=len(ranked)
for label,qs in [('好',(.01,.03,.05)),('中',(.45,.50,.55)),('差',(.99,.97,.95))]:
 for q in qs:
  center=int(round(q*(N-1)));candidates=sorted(range(N),key=lambda i:(abs(i-center),i));row=next(ranked[i] for i in candidates if int(ranked[i]['frame']) not in selected);add(row['frame'],f'双腕最大误差按升序约 {q:.0%} 分位（{label}）')
assert len(selected)==12
out=[]
for frame,reasons in selected.items():
 x=by[frame];entry={'frame':frame,'reasons':reasons,'left_no_observation':frame in noobs,'wrists':{s:x['sides'][s]['wrist_error_cm'] for s in ('left','right')},'hands':{mode:hands[mode][frame] for mode in ('v0','two_stage')}};out.append(entry)
a.output.write_text(json.dumps({'selection_metric':'max of left and right wrist Euclidean distance to DFM, same for v0 and two-stage; visualization-only, selected after evaluation','selected':out},ensure_ascii=False,indent=2)+'\n');print('selected',[(x['frame'],x['reasons'][0],x['wrists']) for x in out])
