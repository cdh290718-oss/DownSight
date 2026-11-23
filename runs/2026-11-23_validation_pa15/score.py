"""Re-score previously selected poses with independent 12- and 15-landmark PA fits."""
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PRIOR = ROOT/'runs/2026-11-23_fullbody_joint_candidates/evidence'
BANK = ROOT/'outputs/fullbody_joint_candidates_20261123'
sys.path.insert(0, str(ROOT/'runs/2026-11-19_rw_frozen200_free12'))
import measure as ref


def main():
    torch.set_num_threads(1)
    val = ROOT/'outputs/stereo_wrist_candidates_20261123'
    hashes = ref.read(PRIOR/'validation_search.json')
    previous = ref.read(PRIOR/'validation_results.json')
    cfg = val/'validation_fit/protocol.json'
    bank = BANK/'validation_bank.npz'
    choicefile = BANK/'validation_choices.json'
    truthfile = val/'validation_truth.json'
    for path, expected in [(bank,hashes['parameter_sha256']), (choicefile,hashes['choices_sha256']),
                           (truthfile,previous['truth_sha256']), (cfg,hashes['protocol_sha256'])]:
        assert ref.sha(path)==expected
    pack = dict(np.load(bank))
    truth,choices = ref.read(truthfile),ref.read(choicefile)
    assert len(choices)==200 and set(choices)==set(previous['rows'])==set(truth)
    assert ref.sha(cfg)==ref.read(PRIOR/'test_search.json')['protocol_sha256']
    model,_ = ref.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    ratios = ref.read(cfg)['landmark_ratios']
    obs = ref.common.OBS
    methods = ('baseline','wrist','fixed','joint')
    metrics = ('pa12_mm','mapped15_pa_mm','legacy15_pa_mm')
    max_pa12_gap = 0.
    original_file = val/'validation_fit/budget_20000.npz'
    assert ref.sha(original_file)==hashes['baseline_sha256']
    original = dict(np.load(original_file))
    rows = {}
    for pair,choice in choices.items():
        target = np.asarray(truth[pair]['gt'])
        assert target.shape==(15,3) and np.isfinite(target).all()
        local = [0,1,choice['fixed'],choice['best']]
        scores = {}
        for i in set(local):
            native = ref.full_joints(pack,layer,choice['offset']+i)
            legacy = native[ref.common.NATIVE_IDS]
            mapped = legacy.copy()
            for index,a,b,k in zip((0,10,14),(15,8,7),(12,11,10),ratios):
                mapped[index] = native[a]+k*(native[b]-native[a])
            assert np.array_equal(mapped[obs],legacy[obs])
            arrays = [(legacy[obs],target[obs]), (mapped,target), (legacy,target)]
            scores[i] = {key:ref.joint_errors(pred,gt)[1] for key,(pred,gt) in zip(metrics,arrays)}
        by_method = dict(zip(methods,[scores[i] for i in local]))
        for method,values in by_method.items():
            gap = abs(values['pa12_mm']-previous['rows'][pair]['scores'][method]['pa'])
            max_pa12_gap = max(gap,max_pa12_gap)
            assert gap<1e-6
        for field in ('global0','delta','body_pose','translation','betas'):
            assert np.array_equal(pack[field][choice['offset']],original[field][choice['index']])
        assert truth[pair]['subject']==previous['rows'][pair]['subject']
        rows[pair] = dict(subject=truth[pair]['subject'],scores=by_method)
    def aggregate(group):
        return {m:{k:float(np.mean([r['scores'][m][k] for r in group])) for k in metrics} for m in methods}
    subjects = {}
    for subject in sorted({r['subject'] for r in rows.values()}):
        group = [r for r in rows.values() if r['subject']==subject]
        assert len(group)==50
        subjects[subject] = dict(n=len(group),means=aggregate(group))
    summary = aggregate(list(rows.values()))
    print(summary,subjects,flush=True)
    ref.write(RUN/'evidence/results.json',dict(pairs=len(rows),summary=summary,subjects=subjects,rows=rows,
              max_pa12_archive_gap_mm=max_pa12_gap,official_names=ref.common.NAMES,observed_indices=obs,ratios=ratios,
              input_hashes={p.name:ref.sha(p) for p in (bank,choicefile,truthfile,cfg,original_file)},
              baseline='1c5904b247534dc25b3fa6835e8b0df8f432593a',exit_status=0))

if __name__=='__main__':
    main()
