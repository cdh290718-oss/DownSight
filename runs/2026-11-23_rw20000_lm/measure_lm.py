"""Score LM and the linked report's Adam column after LM is complete."""
import sys
from pathlib import Path
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0,str(ROOT/'runs/2026-11-23_rw20000_wrist_depth'))
from score import ref, summarize, align, OUT as PREVIOUS
from fit_range import base

OUT = ROOT/'outputs/rw20000_lm_20261123'
KEYS = ('mpjpe_mm','rmse_mm','pa_mpjpe_mm','wrist_rmse_mm','nonwrist_mpjpe_mm','gt_rmse_px')


def evaluate(item, params, i, layer, truths, images, ids2, cameras, adapters):
    pair = item['pair']
    assert [p['native_index'] for p in item['points']]==np.array(ref.common.NATIVE_IDS)[ref.common.OBS].tolist()
    pred = ref.full_joints(params,layer,i)[ref.common.NATIVE_IDS][ref.common.OBS]
    gt = np.asarray(truths[pair]['gt'])[ref.common.OBS]
    distance = np.linalg.norm(pred-gt,axis=1)*1000
    pa = np.linalg.norm(align(pred,gt)-gt,axis=1)*1000
    wrist = np.array([p['group']=='wrist' for p in item['points']])
    views = ref.pixel_errors(item,pred,images[pair],ids2,cameras,adapters)
    return dict(joint_mm=distance.tolist(),pa_mm=pa.tolist(),wrist_mm=distance[wrist].tolist(),
                nonwrist_mm=distance[~wrist].tolist(),n2=sum(v['n'] for v in views),
                gt_sq_px=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views))


def main():
    torch.set_num_threads(1)
    fit = ref.read(RUN/'evidence/fit.json')
    assert fit['exit_status']==0 and fit['feasible']==100
    exec_info = ref.read(base.RUN/'evidence/execution.json')
    ref_file = ref.SOURCE/'reference.json'
    prior = ref.read(ROOT/'runs/2026-11-23_rw20000_wrist_depth/evidence/fit_results.json')
    assert ref.sha(ref_file)==exec_info['reference_sha256']
    assert ref.sha(ref.OUT/'budget_20000.npz')==exec_info['params_sha256']
    assert ref.sha(PREVIOUS/'budget_20000.npz')==prior['params_sha256']
    assert ref.sha(OUT/'lm.npz')==fit['params_sha256']
    truths = ref.read(ref_file)
    images, ids2 = ref.reference_images()
    model,camera = ref.common.setup_model()
    others = (model.layer['neutral'],truths,images,ids2,[camera(p) for p in ref.camera_files()],
              [ref.read(p)['input_adapter'] for p in ref.camera_files()])
    results = {'adam':{},'lm':{}}
    all_items = ref.read(ref.OUT/'prepared.json')['items']
    lm_items = [r for r in all_items if r['condition']=='rw20000']
    assert [r['pair'] for r in lm_items]==[r['pair'] for r in fit['rows']]
    for method,items,file,cond in [
        ('adam',all_items,ref.OUT/'budget_20000.npz','rw20000'),
        ('adam',ref.read(PREVIOUS/'prepared.json')['items'],PREVIOUS/'budget_20000.npz','baseline'),
        ('lm',lm_items,OUT/'lm.npz','rw20000')]:
        with np.load(file) as z:
            params=dict(z)
        assert not params['betas'].any() and not params['body_pose'][:,[11,14]].any()
        for i,item in enumerate(items):
            if item['condition']==cond:
                results[method][item['pair']]=evaluate(item,params,i,*others)
    assert set(results['lm'])==set(results['adam']) and len(results['lm'])==100
    sums={m:summarize(list(rows.values())) for m,rows in results.items()}
    assert all(abs(sums['adam'][k]-prior['summary']['all']['baseline'][k])<1e-6 for k in KEYS)
    summary={m:{k:s[k] for k in KEYS} for m,s in sums.items()}
    summary['lm_minus_adam']={k:summary['lm'][k]-summary['adam'][k] for k in KEYS}
    pairs=[]
    for pair in results['lm']:
        row={'pair':pair,'subject':truths[pair]['subject']}
        for m in results:
            data=results[m][pair]
            row[m]=dict(summarize([data]),gt_sq_px=data['gt_sq_px'],n2=data['n2'])
        pairs.append(row)
    by_subject={}
    for subject in sorted({r['subject'] for r in pairs}):
        selected=[r['pair'] for r in pairs if r['subject']==subject]
        by_subject[subject]={m:summarize([results[m][p] for p in selected]) for m in results}
    ref.write(RUN/'evidence/scores.json',dict(summary=summary,rows=pairs,subjects=by_subject,
              reference_sha256=exec_info['reference_sha256'],manifest_sha256=exec_info['manifest_sha256'],
              adam_params_sha256=[exec_info['params_sha256'],prior['params_sha256']],exit_status=0))
    print(summary,flush=True)


if __name__=='__main__':
    main()
