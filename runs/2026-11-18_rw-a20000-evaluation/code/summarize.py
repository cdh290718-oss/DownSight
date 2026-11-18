"""Aggregate paired derived errors without reading body GT or model parameters."""
import hashlib
import json
from pathlib import Path
import numpy as np

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]
def read(p):return json.loads(p.read_text())
def main():
    rows=read(RUN/'evidence/results.json')['records'];groups={}
    for condition in ['baseline','rw5000','rw20000']:
        rr=[r for r in rows if r['condition']==condition]
        ok=[r for r in rr if r['status']=='ok'];assert len(rr)==10
        absolute=np.array([r['absolute_per_joint_mm'] for r in ok]);pa=np.array([r['pa_per_joint_mm'] for r in ok])
        groups[condition]=dict(success=len(ok),total=len(rr),mpjpe_mm=float(absolute.mean()),pa_mpjpe_mm=float(pa.mean()),
            rmse_mm=float(np.sqrt(np.mean(absolute**2))),convergence_pass=sum(r['convergence_pass'] for r in ok),
            per_joint_mpjpe_mm=absolute.mean(0).tolist(),per_joint_pa_mm=pa.mean(0).tolist(),
            per_pair={r['pair']:dict(mpjpe_mm=r['absolute_mean_mm'],pa_mpjpe_mm=r['pa_mean_mm']) for r in ok})
    old=read(ROOT/'runs/2026-11-17_rw-a5000-evaluation/evidence/results.json')['records']
    oldmap={r['pair']:r['absolute_mean_mm'] for r in old if r['condition']=='rw5000'}
    repeat={pair:abs(v['mpjpe_mm']-oldmap[pair]) for pair,v in groups['rw5000']['per_pair'].items()}
    paired={}
    for reference in ['baseline','rw5000']:
        difference={p:groups[reference]['per_pair'][p]['mpjpe_mm']-v['mpjpe_mm'] for p,v in groups['rw20000']['per_pair'].items()}
        paired[reference]=dict(new_lower=sum(v>0 for v in difference.values()),new_higher=sum(v<0 for v in difference.values()),
            improvement_mm=difference,mean_improvement_mm=groups[reference]['mpjpe_mm']-groups['rw20000']['mpjpe_mm'])
    errors=read(RUN/'evidence/2d_errors.json')['records'];paired2d={}
    for dataset in sorted({r['dataset'] for r in errors}):
        models={v:{r['id']:r for r in errors if r['dataset']==dataset and r['version']==v} for v in ['original','rw5000','rw20000']}
        assert set(models['original'])==set(models['rw5000'])==set(models['rw20000'])
        gains={}
        for ident,row in models['rw20000'].items():
            older=models['rw5000'][ident]
            assert [x is None for x in row['per_joint_error_px']]==[x is None for x in older['per_joint_error_px']]
            gains[ident]=float(np.mean([x for x in older['per_joint_error_px'] if x is not None])-np.mean([x for x in row['per_joint_error_px'] if x is not None]))
        paired2d[dataset]=dict(images=len(gains),new_lower=sum(v>0 for v in gains.values()),new_higher=sum(v<0 for v in gains.values()),
            unit='image mean over valid joints; views/nearby frames are not independent trials')
    out=dict(command='python code/summarize.py',exit_status=0,body3d=groups,paired_3d=paired,paired_2d=paired2d,
        old_rw5000_reproduction_max_absolute_mpjpe_difference_mm=max(repeat.values()),old_rw5000_reproduction_differences_mm=repeat,
        body_result_sha256=hashlib.sha256((RUN/'evidence/results.json').read_bytes()).hexdigest(),
        scope='body3d on prior S9 ten-pair diagnostic; annotation-derived Head/calibration/masks, fixed mean shape, heel proxies; not independent benchmark')
    (RUN/'evidence/summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
