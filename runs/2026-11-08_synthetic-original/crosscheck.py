"""Cross-check body joint errors with the dataset author's evaluation script."""
import json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/synthetic_original_20261108';read=lambda p:json.loads(p.read_text())
d=read(RUN/'evidence/metrics.json');checks=[]
records={r['id']:r for r in read(OUT/'inputs.json')['records']}
for row in d['views']:
    if row['execution_exit']:
        checks.append(dict(side=row['side'],skipped=True,reason='original pipeline failed'));continue
    cfg=read(Path(records[row['side']]['config']));frame=Path(cfg['output'])/'frames'/row['side']
    pred=frame/'fused/body_params.npz'
    if not pred.exists():pred=frame/'body/params.npz'
    p=subprocess.run([sys.executable,str(OUT/'dataset/evaluate_joints.py'),'--prediction',str(pred),'--side',row['side']],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    upstream=json.loads(p.stdout);(RUN/'evidence'/f'{row["side"]}_dataset_evaluation.json').write_text(json.dumps(upstream,indent=2)+'\n')
    group=row['stages']['final']['groups']['body25']
    a=abs(upstream['absolute_mean_mm']-group['absolute_mm']['mean']);b=abs(upstream['pelvis_aligned_mean_mm']-group['pelvis_aligned_mm']['mean'])
    assert a<.001 and b<.001,(row['side'],a,b)
    checks.append(dict(side=row['side'],command='<BODY_PYTHON> <DATASET>/evaluate_joints.py --prediction <BODY_PARAMS> --side '+row['side'],exit_status=p.returncode,absolute_mean_difference_mm=a,pelvis_aligned_mean_difference_mm=b))
(RUN/'evidence/crosscheck.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
print('Independent dataset evaluator agrees for every completed view')
