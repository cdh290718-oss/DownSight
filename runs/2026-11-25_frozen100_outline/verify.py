"""Check cold-replay identity and the paired experiment provenance."""
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from shared_run import RUN,ROOT,OUT,EV,prev,search

def main():
    current=OUT/'cold/budget_20000.npz'
    archive=prev.base.OUT/'budget_20000.npz'
    done=prev.read(OUT/'cold/fit_complete.json')
    assert done['steps']==20000 and done['exit_status']==0
    indices=[i for i,id_ in enumerate(done['ids']) if id_[1]=='rw20000']
    assert len(indices)==100
    diffs={}
    with np.load(current) as fresh,np.load(archive) as old:
        assert set(fresh.files)==set(old.files)
        for k in fresh.files:
            a,b=fresh[k][indices],old[k][indices]
            diffs[k]=float(np.max(abs(a-b)))
            assert np.array_equal(a,b),(k,diffs[k])
    selections=prev.read(EV/'selections.json')['rows']
    res=prev.read(EV/'results.json')
    assert len(selections)==len(res['rows'])==100
    max_param=max(r['archive_max_parameter_difference'] for r in selections.values())
    assert max_param==0.
    assert all(r['prior_v3_choice']==r['v3_choice'] and r['baseline_sha256'] in (prev.sha(current),prev.sha(archive)) for r in selections.values())
    prior=prev.read(ROOT/'runs/2026-11-24_v3_contact/evidence/test_evaluation.json')['rows']
    max_mpjpe={k:max(abs(res['rows'][p]['methods']['v3'][k+'_mm']-prior[p]['before'][k]) for p in selections) for k in ['pa','nonpa']}
    seg=prev.read(EV/'segmentation.json')
    assert len(seg['eyes'])==200
    for k,r in seg['eyes'].items():assert r['sampled_sha256']==prev.sha(OUT/'seg'/(k+'.npy'))
    mask=Counter(name for r in selections.values() for eye in r['eye_support'] for name in eye)
    counts={key:dict(Counter(r['methods'][key]['choice'] for r in selections.values())) for key in ('matched','outline')}
    original=prev.read(search.RUN/'evidence/test_search.json')
    v2=prev.read(search.RUN/'evidence/test_results.json')['summary']['all']['baseline']
    rec=dict(base_commit='ef1400e241216602251fdf81cb7a346342c86157',head_at_verification=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             dirty_worktree=True,original_search_hashes={k:v for k,v in original.items() if k.endswith('sha256')},
             cold_replay=dict(steps=done['steps'],batch_size=len(done['ids']),rw_frames=len(indices),archived_sha256=prev.sha(archive),
                              replay_sha256=prev.sha(current),rw_field_max_abs_difference=diffs,elapsed_seconds=done['elapsed_seconds']),
             v3_max_parameter_difference=max_param,v3_max_metric_difference_mm=max_mpjpe,paired_frame_counts=counts,eye_channel_support=mask,
             earlier_v2_pa_mm=v2['pa'],earlier_v2_nonpa_mm=v2['mpjpe'],parameters=dict(shape='neutral zero',outline_weight=.008,
             steps=72,probability_threshold=.55,outline_sigma_px=12,max_distance_px=64,stride=4,minimum_area=12,minimum_boundary=6),
             recoveries=[dict(command='bash <RUN>/run.sh refit',exit_status=1,error='FileNotFoundError: cold/prepared.json',
                              fix='copied the verified prepared file into the new private output directory before replay'),
                         dict(command='bash <RUN>/run.sh evaluate',exit_status=1,error='ImportError: gap from an older assess module',
                              fix='bound the diagnostic gap routine to the exact two-frame experiment file')],
             commands=[dict(command=c,exit_status=0) for c in ['bash <RUN>/run.sh segment','bash <RUN>/run.sh refit',
                       'bash <RUN>/workers.sh 12','bash <RUN>/run.sh evaluate','bash <RUN>/run.sh chart','bash <RUN>/run.sh regions','bash <RUN>/run.sh verify']],
             source_sha256={f.name:prev.sha(f) for f in RUN.iterdir() if f.suffix in ('.py','.sh')},
             outcome_sha256={p:prev.sha(EV/p) for p in ('segmentation.json','selections.json','results.json','regions.json','paired_changes.png')},exit_status=0)
    prev.write(EV/'execution.json',rec)

if __name__=='__main__':main()
