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
    hashes = ref.read(PRIOR/'test_search.json')
    previous = ref.read(PRIOR/'test_results.json')
    cfg = ref.OUT/'protocol.json'
    bank = BANK/'test_bank.npz'
    choicefile = BANK/'test_choices.json'
    truthfile = ref.SOURCE/'reference.json'
    for path, expected in [(bank,hashes['parameter_sha256']), (choicefile,hashes['choices_sha256']),
                           (truthfile,previous['truth_sha256']), (cfg,hashes['protocol_sha256'])]:
        assert ref.sha(path)==expected
    pack = dict(np.load(bank))
    truth,choices = ref.read(truthfile),ref.read(choicefile)
    historical = {r['pair']:r for r in ref.read(ref.RUN/'evidence/results.json')['rows'] if r['model']=='rw20000'}
    assert len(choices)==100 and set(choices)==set(historical)==set(previous['rows'])==set(truth)
    model,_ = ref.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    ratios = ref.read(cfg)['landmark_ratios']
    obs = ref.common.OBS
    methods = ('baseline','wrist','fixed','joint')
    metrics = ('pa12_mm','mapped15_pa_mm','legacy15_pa_mm')
    max_archive_difference = dict.fromkeys(metrics,0.)
    original_file = ref.OUT/'budget_20000.npz'
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
            assert abs(values['pa12_mm']-previous['rows'][pair]['scores'][method]['pa'])<1e-6
        for field in ('global0','delta','body_pose','translation','betas'):
            assert np.array_equal(pack[field][choice['offset']],original[field][choice['index']])
        for key in metrics:
            gap = abs(by_method['baseline'][key]-historical[pair][key])
            max_archive_difference[key] = max(max_archive_difference[key],gap)
        rows[pair] = by_method
    summary = {m:{k:float(np.mean([r[m][k] for r in rows.values()])) for k in metrics} for m in methods}
    print(summary,max_archive_difference,flush=True)
    assert max(max_archive_difference.values())<0.001
    ref.write(RUN/'evidence/results.json', dict(pairs=len(rows),summary=summary,rows=rows,
              max_archive_difference_mm=max_archive_difference,official_names=ref.common.NAMES,observed_indices=obs,ratios=ratios,
              input_hashes={p.name:ref.sha(p) for p in (bank,choicefile,truthfile,cfg)},
              baseline='4f0378c3d6d01e755e7b2850aaa35e499a1696b6',exit_status=0))

if __name__=='__main__':
    main()
