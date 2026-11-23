"""Evaluate saved choices with official 3D/2D references, including a post-hoc diagnostic."""
import sys
from collections import Counter

import numpy as np
import torch
from search import RUN, OUT, CHAINS, prev
from score_candidates import summarize

KEYS = ['baseline','wrist','fixed','joint','oracle']
METRICS = ['mpjpe','rmse','pa','wrist_rmse','other10','px']


def measure(split):
    torch.set_num_threads(1)
    search = prev.read(RUN/('evidence/'+split+'_search.json'))
    path = OUT/(split+'_bank.npz')
    choices = OUT/(split+'_choices.json')
    assert prev.sha(path)==search['parameter_sha256'] and prev.sha(choices)==search['choices_sha256']
    poses = dict(np.load(path))
    records = prev.read(choices)
    manifest,folder,truthfile = ((prev.TM,prev.base.OUT,prev.ref.SOURCE/'reference.json') if split=='test' else
                                (prev.VM,prev.VAL,prev.OUT/'validation_truth.json'))
    previous = prev.read(prev.RUN/('evidence/'+split+'_results.json'))
    assert prev.sha(truthfile)==previous['reference_sha256']
    assert prev.sha(manifest)==prev.read(prev.RUN/'evidence/splits.json')['hashes'][split+'_manifest']
    truth = prev.read(truthfile)
    images = prev.pair_images(manifest)
    items = prev.read(folder/'prepared.json')['items']
    model,cam = prev.ref.common.setup_model()
    cams = [cam(p) for p in prev.ref.camera_files()]
    adapters = [prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    ids2 = prev.mapping()
    obs = prev.ref.common.OBS
    names = [prev.ref.common.NAMES[k] for k in obs]
    natives = np.asarray(prev.ref.common.NATIVE_IDS)[obs]
    rows = {}
    for key,r in records.items():
        item = items[r['index']]
        im = images[int(key[-3:])-1]
        assert key==item['pair'] and im[0]['subject']==truth[key]['subject']
        target = np.asarray(truth[key]['gt'])[obs]
        start = r['offset']
        predictions = [prev.ref.full_joints(poses,model.layer['neutral'],start+j)[natives] for j in range(len(r['names']))]
        errs = 1000*np.linalg.norm(np.array(predictions)-target,axis=-1)
        oracle = int(errs.mean(1).argmin())
        indices = [0,1,r['fixed'],r['best'],oracle]
        assert r['loss'][r['best']] <= min(r['loss'][:3])
        scores = {}
        for method,j in zip(KEYS,indices):
            pred = predictions[j]
            views = prev.ref.pixel_errors(item,pred,im,ids2,cams,adapters)
            scores[method] = dict(mm=errs[j].tolist(),pa_mm=(1000*np.linalg.norm(prev.align(pred,target)-target,axis=1)).tolist(),
                              n_px=sum(v['n'] for v in views),px_sq=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views),
                              det_sq=sum(v['sum_sq_px2']['fit_vs_detection'] for v in views),loss=r['loss'][j])
        def mask(j):
            from scipy.spatial.transform import Rotation as R
            changed = []
            for s,e,w in CHAINS:
                b0,b1 = [poses['body_pose'][start+k,[s-1,e-1]] for k in (1,j)]
                changed.append(bool((R.from_rotvec(b0).inv()*R.from_rotvec(b1)).magnitude().max()>np.deg2rad(.1)))
            return changed
        rows[key] = dict(subject=truth[key]['subject'],choice=r['names'][r['best']],oracle=r['names'][oracle],
                         changed=mask(r['best']),scores=scores)
    def summary(keys):
        result = {}
        for k in KEYS:
            rr = [rows[p]['scores'][k] for p in keys]
            result[k] = dict(summarize(rr),loss=float(np.mean([v['loss'] for v in rr])),
                             det_px=float(np.sqrt(sum(v['det_sq'] for v in rr)/sum(v['n_px'] for v in rr))),
                             joint_mm=dict(zip(names,np.mean([v['mm'] for v in rr],axis=0).tolist())))
        return result
    subsets = dict(all=list(rows))
    for sub in sorted({r['subject'] for r in rows.values()}):
        subsets[sub] = [k for k,r in rows.items() if r['subject']==sub]
    if split=='test':
        subsets['without_prior_outliers'] = [k for k in rows if k not in ('pair003','pair004')]
    summaries = {k:summary(v) for k,v in subsets.items()}
    for k,old in [('baseline','baseline'),('wrist','objective')]:
        for metric in METRICS:
            assert abs(summaries['all'][k][metric]-previous['summary']['all'][old][metric])<1e-6
    def paired(key):
        scores = rows[key]['scores']
        return float(np.mean(scores['joint']['mm'])-np.mean(scores['wrist']['mm']))
    diffs = np.array([paired(k) for k in rows])
    diag = dict(chain_order=['left_arm','right_arm','left_leg','right_leg'],
                changed_chains_histogram=dict(Counter(sum(r['changed']) for r in rows.values())),
                changed_by_chain=np.sum([r['changed'] for r in rows.values()],axis=0).tolist(),
                selection=dict(Counter(r['choice'] for r in rows.values())),
                lower_loss=sum(r['scores']['joint']['loss']<r['scores']['wrist']['loss']-1e-8 for r in rows.values()),
                mpjpe_improved=int(sum(diffs < -.01)),mpjpe_worsened=int(sum(diffs>.01)),
                mpjpe_tied=int(sum(abs(diffs)<=.01)),median_delta_mm=float(np.median(diffs)),
                extreme_pairs={k:paired(k) for k in sorted(rows,key=paired)[:3]+sorted(rows,key=paired)[-3:]})
    # Sufficient per-pair statistics for the six metrics, without raw poses, images or keypoints.
    for r in rows.values():
        r['scores'] = {k:dict(summarize([v]),loss=v['loss'],n_px=v['n_px'],det_sq=v['det_sq']) for k,v in r['scores'].items()}
    prev.write(RUN/('evidence/'+split+'_results.json'),dict(summary=summaries,diagnostics=diag,rows=rows,
               truth_sha256=prev.sha(truthfile),manifest_sha256=prev.sha(manifest),choices_sha256=prev.sha(choices),exit_status=0))
    print(split,diag,flush=True)


if __name__=='__main__':
    measure(sys.argv[1])
