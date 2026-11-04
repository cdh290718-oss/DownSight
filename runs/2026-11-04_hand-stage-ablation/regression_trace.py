"""Preserve the largest C hand-depth regression, including optimizer bounds."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/hand_stage_ablation_20261104'
read=lambda p:json.loads(p.read_text())
d=read(RUN/'evidence/metrics.json')
worst=max(d['comparisons']['C']['changes'],key=lambda q:q['delta_3d_m'])
frame=next(r for r in d['frames'] if r['pair']==worst['pair'] and r['camera']==worst['camera'])
result=dict(worst=worst,frame=frame['frame'],methods={})
for group in ['A','B','C']:
    rec=next(r for r in read(OUT/group/'inputs.json')['records'] if r['id']==frame['frame'])
    root=Path(read(Path(rec['config']))['output'])/'frames'/rec['id']
    pose=read(root/'body_fit/automatic/refinement_report.json');guided=read(root/'guided_body/optimization.json')
    result['methods'][group]=dict(stages={s:{'pelvis_z_m':v['pelvis_z_m'],'shoulder_rmse_px':v['shoulder_rmse_px'],'hand':v['hands'][worst['side']]} for s,v in frame['methods'][group]['stages'].items()},
        pose_limits=pose['joints_at_increment_limit'],pose_increments_deg=pose['pose_increment_deg'],guided_steps=guided['actual_steps'],guided_increments_deg=guided['acceptance']['body_delta_deg'],guided_trace=guided['trace'],guided_cpu_step_fraction=guided['cpu_feasibility_step_fraction'])
(RUN/'evidence/largest_C_regression.json').write_text(json.dumps(result,indent=2)+'\n')
print(worst)
