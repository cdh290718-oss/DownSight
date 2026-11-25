"""Validate paired starts, geometry outcomes, exclusion support and public evidence."""
import subprocess
from geometry_inputs import RUN,EV,OUT,PREV,prev,search
from collections import Counter
import numpy as np

def main():
    d=prev.read(EV/'results.json');old=prev.read(PREV/'evidence/results.json')
    selections=prev.read(EV/'selections.json')
    for m in ['v3','matched','outline']:
        for k in ('pa_mm','nonpa_mm'):
            assert abs(d['summary'][m][k]-old['summary'][m][k])<1e-8
    assert len(d['rows'])==100 and set(d['rows'])==set(selections)
    assert d['support']['no_hand']['retained_samples']==0
    assert d['support']['palm']==d['support']['free_palm']
    assert d['support']['fixed_full']['retained_samples']>=d['support']['palm']['retained_samples']
    p=prev.read(EV/'replay.json')
    assert not any(p['max_abs_difference'].values())
    for m in ['fixed_full','palm','no_hand']:
        assert max(r['methods'][m]['finger_max_delta_rad'] for r in d['rows'].values())==0
    old_regions=prev.read(PREV/'evidence/regions.json')
    for m in ['v3','matched','outline']:
        assert d['summary'][m]['self_frames']==old_regions['any_within_hand_frames'][m]
    diffs={}
    for mode in ['fixed_full','palm','no_hand','free_palm']:
        diffs[mode]={}
        for base in ['v3','outline','fixed_full']:
            metrics={}
            for k in ['pa_mm','nonpa_mm']:
                a=np.array([r['methods'][mode][k]-r['methods'][base][k] for r in d['rows'].values()])
                metrics[k]=dict(mean=float(a.mean()),median=float(np.median(a)),improved=int((a<-.01).sum()),worse=int((a>.01).sum()))
            diffs[mode][base]=metrics
    previous=prev.read(PREV/'evidence/execution.json')
    previous_hashes={name:prev.sha(PREV/'evidence'/name) for name in ('selections.json','segmentation.json','results.json','regions.json')}
    prev.write(EV/'execution.json',dict(base_commit='b8b5dde72b716e0d6f14ae12ae16b3bb89f20234',head_at_verification=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_worktree=True,device='CPU',prior_evidence_sha256=previous_hashes,input_sha256=previous['original_search_hashes'],
        source_sha256={p.name:prev.sha(p) for p in RUN.iterdir() if p.suffix in ('.sh','.py')},
        evidence_sha256={p:prev.sha(EV/p) for p in ['results.json','selections.json','replay.json','hand_edges.png']},delta=diffs,
        config=dict(steps=72,shape='zero beta',contour_weight=.008,grid_stride_px=4,obs_digit_halo_cells=2,
                    palm_support_cells=2,source_halo_cells=1,min_boundary_samples=6),
        commands=[dict(command='bash <RUN>/'+s,exit_status=0) for s in ['workers.sh 16 fixed','workers.sh 10 free','run.sh replay',
                  'run.sh diagram','run.sh score','run.sh verify']],exit_status=0))

if __name__=='__main__':main()
