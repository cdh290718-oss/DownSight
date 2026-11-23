"""Post-fit evaluation of paired solutions, using official joint positions for scoring."""
import json
import sys
import numpy as np
import torch
from angles import EVIDENCE, OUTPUT, ROOT, features, sha, write

sys.path.insert(0, str(ROOT/'runs/2026-11-23_stereo_wrist_candidates'))
import shared

ARMS = ('baseline', 'rw_prior')


def summarize(rows):
    errors = np.array([r['mm'] for r in rows])
    obs_names = [shared.ref.common.NAMES[k] for k in shared.ref.common.OBS]
    wrists = np.array(['Wrist' in k for k in obs_names])
    return dict(pairs=len(rows), feasible=sum(r['feasible'] for r in rows), converged=sum(r['converged'] for r in rows),
                mpjpe_mm=errors.mean(), pa_mm=np.mean([r['pa_mm'] for r in rows]), rmse_mm=np.sqrt((errors**2).mean()),
                wrist_rmse_mm=np.sqrt((errors[:,wrists]**2).mean()), other10_mm=errors[:,~wrists].mean(),
                px_rmse=np.sqrt(sum(r['px_sq'] for r in rows)/sum(r['view_points'] for r in rows)),
                angle_mae_deg=np.mean([r['angle_mae_deg'] for r in rows]),
                signed_component_mae=np.mean([r['direction_mae'] for r in rows]),
                any_bound_exceeded=sum(r['any_outside'] for r in rows))


def main(split):
    torch.set_num_threads(1)
    fitdir = OUTPUT / split
    manifest = shared.TM if split == 'test' else shared.VM
    collection = 'Test' if split == 'test' else 'Train'
    source = ROOT.parent/'EgoSMPLX/data'/('UnrealEgo_RW_'+collection)/'prepared'
    truthfile = source/'samples.npz'
    expected = shared.read(EVIDENCE/'survey.json')['sources'][collection]['samples_sha256']
    assert sha(truthfile) == expected
    assert sha(manifest) == shared.read(shared.RUN/'evidence/splits.json')['hashes'][split+'_manifest']
    images = shared.pair_images(manifest)
    with np.load(truthfile) as labels:
        xyz = labels['keypoints3d']
        seq, frames, sides = (labels[k] for k in ('source_sequence','frame_index','camera_side'))
    names_gt = json.loads((source/'metadata.json').read_text())['train_joint_names']
    items = shared.read(fitdir/'prepared.json')['items']
    diag = shared.read(fitdir/'diagnostic_20000.json')
    assert all(diag['selected_feasible']) and min(diag['selected_step']) >= 0
    selected_file = fitdir/'budget_20000.npz'
    assert sha(selected_file) == shared.read(EVIDENCE/(split+'_fit.json'))['selected_fit_sha256']
    selected = dict(np.load(selected_file))
    assert np.array_equal(diag['selected_step'], selected['selected_step'])
    assert not selected['betas'].any() and not selected['body_pose'][:,[11,14]].any()
    model, camera = shared.ref.common.setup_model()
    layer = model.layer['neutral']
    cams = [camera(p) for p in shared.ref.camera_files()]
    adapters = [shared.read(p)['input_adapter'] for p in shared.ref.camera_files()]
    obs = shared.ref.common.OBS
    native = np.asarray(shared.ref.common.NATIVE_IDS)[obs]
    names = [shared.ref.common.NAMES[k] for k in obs]
    lookup = {name: i for i,name in enumerate(names)}
    prior = shared.read(EVIDENCE/'prior.json')
    ids = shared.mapping()
    rows = {}
    for i, item in enumerate(items):
        pair, arm = item['pair'], item['condition']
        assert diag['ids'][i] == [pair,arm,'2d']
        pair_images = images[int(pair[-3:])-1]
        left = pair_images[0]
        label_index = left['source_index']
        assert (seq[label_index], int(frames[label_index]), sides[label_index]) == (left['sequence'],left['frame'],'left')
        assert left['subject'] == left['sequence'].split('/')[2]
        target = xyz[label_index,[names_gt.index(n) for n in names]].astype(float)
        pred = shared.ref.full_joints(selected, layer, i)[native]
        views = shared.ref.pixel_errors(item, pred, pair_images, ids, cams, adapters)
        f, gt = features(torch.from_numpy(np.stack([pred, target])), lookup).numpy()
        mm = np.linalg.norm(pred-target, axis=1)*1000
        row = rows.setdefault(pair, dict(subject=left['subject']))
        row[arm] = dict(mm=mm.tolist(), pa_mm=float(np.linalg.norm(shared.align(pred,target)-target,axis=1).mean()*1000),
            view_points=sum(v['n'] for v in views), px_sq=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views),
            feasible=diag['selected_feasible'][i], converged=diag['convergence_pass'][i], selected_step=diag['selected_step'][i],
            loss=float(selected['loss'][i]), any_outside=bool(((f<prior['lower'])|(f>prior['upper'])).any()),
            angle_mae_deg=float(abs(f[:,0]-gt[:,0]).mean()), direction_mae=float(abs(f[:,1]-gt[:,1]).mean()))
    assert len(rows) == len(images) and all(set(x)=={'subject',*ARMS} for x in rows.values())
    groups = {'all': list(rows)}
    for subject in sorted({r['subject'] for r in rows.values()}):
        groups[subject] = [k for k,r in rows.items() if r['subject']==subject]
    summary = {g: {arm:summarize([rows[k][arm] for k in keys]) for arm in ARMS} for g,keys in groups.items()}
    diff = {k:float(np.mean(r['rw_prior']['mm'])-np.mean(r['baseline']['mm'])) for k,r in rows.items()}
    delta = np.array(list(diff.values()))
    paired = dict(tolerance_mm=.01, improved=int(sum(delta<-.01)), worsened=int(sum(delta>.01)),
                  tied=int(sum(abs(delta)<=.01)), mean_mm=delta.mean(), median_mm=np.median(delta),
                  best=min(diff,key=diff.get), worst=max(diff,key=diff.get))
    write(EVIDENCE/(split+'_evaluation.json'), dict(command='bash run.sh score '+split, exit_status=0, joint_order=names,
          summary=summary, paired=paired, truth_sha256=expected, manifest_sha256=sha(manifest),
          selected_sha256=sha(selected_file), rows=rows))
    print(split, summary['all'], paired, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
