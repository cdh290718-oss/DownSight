"""Check aggregation, sealed selections, and historical metric agreement before release."""
import json
from pathlib import Path
import subprocess
from inputs import RUN, ROOT, EV, prev, search, FINGERS
from fit_probe import STEPS, WEIGHT
from outline import STRIDE, SIGMA_PX


def main():
    result = prev.read(EV/'results.json')
    selection = prev.read(EV/'selections.json')
    eyes = prev.read(EV/'segmentation.json')['eyes']
    rows = result['rows']
    assert len(rows)==200 and set(rows)==set(selection) and len(eyes)==2*len(rows)
    assert max(result['baseline_max_error_difference_mm'].values())<.001
    assert not result['baseline_crossing_count_disagreements']
    assert result['support']['fixed_full']['retained_hand_samples']>=result['support']['palm']['retained_hand_samples']
    for mode,r in result['summary'].items():
        assert r['frames']==len(rows) and r['finger_max_delta_rad']==0
        for metric in ['pa_mm','nonpa_mm','full_outline_px']:
            assert abs(r[metric]-sum(x['methods'][mode][metric] for x in rows.values())/len(rows))<1e-8
    for comparison,d in result['paired'].items():
        assert sum(d['crossings'].values())==len(rows)
        for metric in ['pa_mm','nonpa_mm','full_outline_px']:
            assert sum(d[metric][k] for k in ['improved','worsened','within_0p01'])==len(rows)
    old = ROOT/'runs/2026-11-24_v3_contact/evidence'
    files = [search.RUN/'evidence/validation_search.json',old/'verification.json',old/'validation_evaluation.json',
             FINGERS/'evidence/results.json',FINGERS/'fit.py',FINGERS/'hand_edges.py',
             FINGERS.parent/'2026-11-25_frozen100_outline/compare.py']
    base = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert base=='a8e4a94ee1c949dee7a7d322f78956ec9a3f70ea'
    commands = ['run.sh segment','workers.sh 16 variants','run.sh score','run.sh verify']
    prev.write(EV/'execution.json',dict(
        base_commit=base,dirty_worktree=True,split='validation',device='single CUDA GPU segmentation / CPU fit and scoring',
        input_archive={k:v for k,v in prev.read(files[0]).items() if k.endswith('_sha256')},
        historical_sources={str(p.relative_to(ROOT)):prev.sha(p) for p in files},
        run_sources={p.name:prev.sha(p) for p in RUN.iterdir() if p.suffix in ('.py','.sh')},
        outputs={p:prev.sha(EV/p) for p in ['results.json','selections.json','segmentation.json']},
        config=dict(steps=STEPS,weight=WEIGHT,grid_stride_px=STRIDE,outline_scale_px=SIGMA_PX,
                    shape='10 zero betas',body_observations=len(prev.ref.common.OBS),fixed_finger_joints=30,
                    includes_contact_loss=False),
        commands=[dict(command='bash <RUN>/'+c,exit_status=0) for c in commands],exit_status=0))


if __name__=='__main__':
    main()
