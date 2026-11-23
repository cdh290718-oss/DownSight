"""Factor the fixed-trunk solution into arms and legs, then score locked predictions."""
import sys
from pathlib import Path

import numpy as np
import torch
from search import RUN, OUT, prev
from score_candidates import summarize


def isolate(split):
    torch.set_num_threads(1)
    prev.base.fit.camera_files = prev.ref.camera_files
    folder = prev.base.OUT if split=='test' else prev.VAL
    bank = OUT/(split+'_bank.npz')
    recfile = OUT/(split+'_choices.json')
    search = prev.read(RUN/('evidence/'+split+'_search.json'))
    assert prev.sha(bank)==search['parameter_sha256'] and prev.sha(recfile)==search['choices_sha256']
    poses,rec = dict(np.load(bank)),prev.read(recfile)
    cfg = prev.read(folder/'protocol.json')
    items = prev.read(folder/'prepared.json')['items']
    model,camera = prev.ref.common.setup_model()
    mask = [15,16,17,18]
    rows = {}
    packed = []
    for key,r in rec.items():
        start = r['offset']
        wrist = {k:arr[start+1].copy() for k,arr in poses.items()}
        four = poses['body_pose'][start+2]
        assert r['names'][2]=='fixed'
        arm = wrist['body_pose'].copy()
        arm[mask] = four[mask]
        leg = four.copy()
        leg[mask] = wrist['body_pose'][mask]
        p = prev.base.trial.TorsoOnly([dict(items[r['index']],condition='baseline')],cfg,model,camera)
        with torch.no_grad():
            b = torch.from_numpy(np.stack([wrist['body_pose'],arm,leg,four])).float()
            seed,d,t,beta = [torch.as_tensor(wrist[k]).float().expand(4,-1) for k in ['global0','delta','translation','betas']]
            xyz,rh = p.fk(b,beta,seed,d,t)
            p.bodyvar = b[:,p.body_ids]
            loss,ok,_ = p.objective((xyz,rh,beta))
        assert ok.all() and abs(loss[0].item()-r['loss'][1])<1e-6
        assert abs(loss[3].item()-r['loss'][2])<1e-6
        # Independent chains share a fixed torso. Check the original objective's additivity.
        assert abs((loss[1]+loss[2]-loss[0]-loss[3]).item())<2e-6
        keep = bool(loss[1]<loss[0])
        if keep:
            wrist['body_pose'] = arm
        packed.append(wrist)
        rows[key] = dict(arms_chosen=keep,component_losses=loss.tolist())
    path = OUT/(split+'_arms.npz')
    np.savez_compressed(path,pair=list(rows),**{k:np.stack([r[k] for r in packed]) for k in packed[0]})
    prev.write(OUT/(split+'_arms.json'),rows)
    print(split,len(rows),prev.sha(path),flush=True)


def score(split):
    torch.set_num_threads(1)
    manifest,folder,truthfile = ((prev.TM,prev.base.OUT,prev.ref.SOURCE/'reference.json') if split=='test' else
                                 (prev.VM,prev.VAL,prev.OUT/'validation_truth.json'))
    model,cam = prev.ref.common.setup_model()
    cams = [cam(p) for p in prev.ref.camera_files()]
    adapters = [prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    ids2 = prev.mapping()
    pairs,truth = prev.pair_images(manifest),prev.read(truthfile)
    items = prev.read(folder/'prepared.json')['items']
    rec = prev.read(OUT/(split+'_choices.json'))
    path = OUT/(split+'_arms.npz')
    pack = dict(np.load(path))
    refs = prev.read(RUN/('evidence/'+split+'_results.json'))
    rows = {}
    for i,key in enumerate(pack['pair']):
        item = items[rec[key]['index']]
        full = prev.ref.full_joints(pack,model.layer['neutral'],i)
        pred = full[prev.ref.common.NATIVE_IDS][prev.ref.common.OBS]
        gt = np.asarray(truth[key]['gt'])[prev.ref.common.OBS]
        views = prev.ref.pixel_errors(item,pred,pairs[int(key[-3:])-1],ids2,cams,adapters)
        rows[key] = dict(mm=(1000*np.linalg.norm(pred-gt,axis=1)).tolist(),
                         pa_mm=(1000*np.linalg.norm(prev.align(pred,gt)-gt,axis=1)).tolist(),
                         px_sq=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views),n_px=sum(v['n'] for v in views))
    frame_stats = {k:summarize([v]) for k,v in rows.items()}
    fixed = {p:r['scores']['fixed']['mpjpe'] for p,r in refs['rows'].items()}
    leg_diff = {p:fixed[p]-r['mpjpe'] for p,r in frame_stats.items() if abs(fixed[p]-r['mpjpe'])>.01}
    gain = {p:r['scores']['wrist']['mpjpe']-r['scores']['joint']['mpjpe'] for p,r in refs['rows'].items()}
    top = max(gain,key=gain.get)
    left = [p for p in rows if p!=top]
    prev.write(RUN/('evidence/'+split+'_ablation.json'),dict(expanded_arms=summarize(list(rows.values())),
             fixed_minus_expanded_arms_mm=leg_diff,largest_gain_pair=top,largest_gain_share=gain[top]/sum(gain.values()),
             without_largest_gain_mpjpe={k:np.mean([refs['rows'][p]['scores'][k]['mpjpe'] for p in left])
                                        for k in ['wrist','fixed','joint']},rows=frame_stats,
             arms_parameters_sha256=prev.sha(path),component_loss_sha256=prev.sha(OUT/(split+'_arms.json')),exit_status=0))


if __name__=='__main__':
    phase,split=sys.argv[1].split('-')
    if phase=='isolate':
        def guard(event,args):
            if event=='open' and args and isinstance(args[0],str):
                name = Path(args[0]).name
                if name in {'reference.json','validation_truth.json','samples.npz','validation.json','test_manifest.json'} or name.endswith('_results.json'):
                    raise PermissionError('Labels excluded from component isolation')
        sys.addaudithook(guard)
        isolate(split)
    elif phase=='score':
        score(split)
    else:
        raise ValueError(phase)
