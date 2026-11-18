"""Summarize fixed-budget coordinate swaps and verify control reproduction."""
import hashlib
import json
from pathlib import Path
import numpy as np

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]
OUT=ROOT/'outputs/rw20000_regression_diagnosis_20261118/d_fit'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
    result=read(RUN/'evidence/results.json');rows=result['records'];assert len(rows)==54
    assert all(r['status']=='ok' and r['selected_feasible'] for r in rows)
    names=result['joint_names'];summary={}
    for condition in sorted({r['condition'] for r in rows}):
        rr=[r for r in rows if r['condition']==condition]
        a=np.array([r['absolute_per_joint_mm'] for r in rr]);b=np.array([r['pa_per_joint_mm'] for r in rr])
        assert np.isfinite(a).all() and np.isfinite(b).all()
        summary[condition]=dict(pairs=len(rr),non_pa_mm=float(a.mean()),pa_mm=float(b.mean()),
            per_joint_non_pa_mm=dict(zip(names,a.mean(0).tolist())),convergence_pass=sum(r['convergence_pass'] for r in rr))
    previous=read(ROOT/'runs/2026-11-18_rw-a20000-evaluation/evidence/results.json')['records']
    repeat=[]
    for condition,ref in [('baseline','rw20000'),('old_all','rw5000')]:
        for row in [r for r in rows if r['condition']==condition]:
            prior=next(r for r in previous if r['condition']==ref and r['pair']==row['pair'])
            delta=abs(row['absolute_mean_mm']-prior['absolute_mean_mm']);assert delta<1e-6
            repeat.append(dict(condition=condition,pair=row['pair'],absolute_difference_mm=delta))
    body07={r['condition']:dict(non_pa_mm=r['absolute_mean_mm'],pa_mm=r['pa_mean_mm'],
                convergence_pass=r['convergence_pass'],per_joint_non_pa_mm=dict(zip(names,r['absolute_per_joint_mm'])))
            for r in rows if r['pair']=='body07'}
    diag=read(RUN/'evidence/diagnostic_2d.json')['records'];joints={}
    for joint in diag[0]['points']:
        joints[joint]={v:float(np.mean([e for r in diag if r['version']==v for e,valid in zip(r['points'][joint]['error_px'],r['points'][joint]['valid']) if valid])) for v in ['rw5000','rw20000']}
    complete=read(OUT/'fit_complete.json');assert complete['exit_status']==0 and not complete['body_gt_read']
    init=read(OUT/'initialization.json')['initialization'];assert len(init)==10 and all(x['rank']==0 for x in init.values())
    base=ROOT/'runs/2026-11-18_rw-a20000-evaluation'
    for n in ['fit.py','geometry.py']:assert (RUN/n).read_bytes()==(base/n).read_bytes()
    protocol=read(RUN/'protocol.json');oldprotocol=read(base/'protocol.json')
    differences={k for k,v in protocol.items() if v!=oldprotocol.get(k)}
    assert differences <= {'baseline_commit','intervention','conditions'}
    write(RUN/'evidence/summary.json',dict(command='python code/summarize.py',exit_status=0,
        conditions=summary,body07=body07,diagnostic_2d_joint_means_px=joints,
        body07_regression_recovered_by_right_knee_ankle_percent=100*(body07['baseline']['non_pa_mm']-body07['old_right_knee_ankle']['non_pa_mm'])/(body07['baseline']['non_pa_mm']-body07['old_all']['non_pa_mm']),
        controls_reproduction=repeat,max_control_difference_mm=max(x['absolute_difference_mm'] for x in repeat),
        all_feasible=True,total_instances=len(rows),total_converged=sum(r['convergence_pass'] for r in rows),
        source_files_unchanged=['fit.py','geometry.py'],protocol_changed_fields=sorted(differences),
        fitting_body_gt_read=False,shape_rank_zero_pairs=10,
        scope='Post-hoc coordinate swap on old diagnostic examples; finite-budget effects, not deployment gains or additive attribution.'))
    write(RUN/'evidence/fit_execution.json',dict(command='PYTHONPATH=<REPO> <BODY_PYTHON> fit.py --threads 2',exit_status=0,completion=complete))
    print(json.dumps(dict(conditions={k:{n:v[n] for n in ['pairs','non_pa_mm','pa_mm','convergence_pass']} for k,v in summary.items()},control_max_difference_mm=max(x['absolute_difference_mm'] for x in repeat))))

if __name__=='__main__':main()
