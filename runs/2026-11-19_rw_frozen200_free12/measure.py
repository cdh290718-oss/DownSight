"""Evaluate selected fits in the original left-camera and image coordinate systems."""
from collections import Counter

import numpy as np
import torch

from run_fit import GOLD, MODELS, OUT, ROOT, RUN, SOURCE, camera_files, common, read, sha, write
from metrics import FIELDS, RMSE, joint_errors
from geometry import rotation


def reference_images():
    manifest = ROOT / 'outputs/rw_test_completion_20261117/test_manifest.json'
    source = read(GOLD / 'evidence/preparation.json')
    assert sha(manifest) == source['manifest_sha256']
    images = {x['id']: x for x in read(manifest)['records']}
    pairs = {x['pair']: [images[i] for i in x['image_ids']] for x in source['pairs']}
    mapping = read(ROOT / 'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping']
    ids = {m['rw_index']: m['sapiens308_id'] for m in mapping}
    return pairs, [ids[j] for j in common.OBS]


def pixel_errors(item, native, images, joint_ids, cameras, adapters):
    views = []
    for v, side in enumerate(('left', 'right')):
        image = images[v]
        assert image['view'] == side
        adapter = adapters[v]
        def to_original(x):
            return (np.asarray(x) - adapter['offset']) / adapter['scale']
        transform = np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
        projected = to_original(cameras[v].world2camera(native @ transform[:3,:3].T + transform[:3,3]))
        detection = to_original([p['xy'][v] for p in item['points']])
        truth = np.asarray(image['keypoints308_xy'])[joint_ids]
        mask = np.asarray([p['valid'][v] for p in item['points']], bool)
        assert np.array_equal(mask, np.asarray(image['keypoints308_supervision_mask'])[joint_ids])
        residuals = (projected - detection, projected - truth, detection - truth)
        views.append(dict(side=side, n=int(mask.sum()), sum_sq_px2={
            k: float(np.square(d[mask]).sum()) for k,d in zip(RMSE, residuals)}))
    return views


def full_joints(params, layer, i):
    def tensor(k):
        return torch.tensor(params[k][i], dtype=torch.float32).reshape(1, -1)
    fixed = {k: torch.zeros(1, n) for k,n in [('left_hand_pose',45),('right_hand_pose',45),
             ('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]}
    with torch.no_grad():
        joints = layer(body_pose=tensor('body_pose'), betas=tensor('betas'),
                       global_orient=tensor('global0'), **fixed).joints[0]
        pivot = joints[:1]
        return ((joints-pivot) @ rotation(tensor('delta'))[0].T + pivot + tensor('translation')).numpy()


def summarize(rows):
    views = [v for row in rows for v in row['views']]
    n = sum(v['n'] for v in views)
    return dict(pairs=len(rows), view_points=n, **{k:float(np.mean([r[k] for r in rows])) for k in FIELDS},
                **{k+'_rmse_px':float(np.sqrt(sum(v['sum_sq_px2'][k] for v in views)/n)) for k in RMSE})


def main():
    items = read(OUT / 'prepared.json')['items']
    diag = read(OUT / 'diagnostic_20000.json')
    done = read(OUT / 'fit_complete.json')
    assert diag['ids'] == done['ids'] and done['exit_status'] == 0
    assert done['protocol_sha256'] == sha(OUT / 'protocol.json')
    source = read(GOLD / 'evidence/preparation.json')
    assert sha(SOURCE / 'reference.json') == source['reference_local_sha256']
    refs = read(SOURCE / 'reference.json')
    images, joint_ids = reference_images()
    model, camera = common.setup_model()
    cams = [camera(p) for p in camera_files()]
    adapters = [read(p)['input_adapter'] for p in camera_files()]
    ratios = read(OUT / 'protocol.json')['landmark_ratios']
    torch.set_num_threads(1)
    rows = []
    with np.load(OUT / 'budget_20000.npz') as data:
        params = dict(data)
    assert not params['body_pose'][:,[11,14]].any() and not params['betas'].any()
    for i,(pair, condition, _) in enumerate(done['ids']):
        row = items[i]
        assert (pair,condition) == (row['pair'],row['condition'])
        assert [p['name'] for p in row['points']] == [common.NAMES[j] for j in common.OBS]
        ref = refs[pair]
        assert all(x['frame']==ref['frame'] and x['sequence']==ref['sequence'] for x in images[pair])
        ok = bool(diag['selected_feasible'][i] and params['selected_step'][i] >= 0)
        r = dict(pair=pair, model='original' if condition=='baseline' else condition,
                 subject=ref['subject'], feasible=ok, converged=diag['convergence_pass'][i],
                 selected_step=diag['selected_step'][i], relative_loss_change=diag['relative_loss_change'][i],
                 max_joint_motion_mm=diag['max_joint_motion_mm'][i])
        rows.append(r)
        if not ok:
            continue
        j = full_joints(params, model.layer['neutral'].cpu().eval(), i)
        legacy = j[common.NATIVE_IDS]
        mapped = legacy.copy()
        for idx,a,b,k in zip((0,10,14),(15,8,7),(12,11,10),ratios):
            mapped[idx] = j[a] + k*(j[b]-j[a])
        gt = np.asarray(ref['gt'])
        scores = joint_errors(legacy[common.OBS], gt[common.OBS]) + joint_errors(legacy, gt) + joint_errors(mapped, gt)
        r.update(zip(FIELDS, scores))
        r['views'] = pixel_errors(row, legacy[common.OBS], images[pair], joint_ids, cams, adapters)

    successful = {m:{r['pair'] for r in rows if r['model']==m and r['feasible']} for m in MODELS}
    paired = set.intersection(*successful.values())
    summary, subjects, all_available = {}, {}, {}
    seed_audit = read(RUN / 'evidence/inputs.json')['rows']
    for m in MODELS:
        records = [r for r in rows if r['model']==m and r['pair'] in paired]
        all_available[m] = summarize([r for r in rows if r['model']==m and r['feasible']])
        summary[m] = summarize(records)
        summary[m].update(feasible=len(successful[m]), converged=sum(r['converged'] for r in records),
                          alternative_seeds=sum(bool(r['stereo_seed_issue']) for r in seed_audit if r['model']==m))
        subjects[m] = {s:summarize([r for r in records if r['subject']==s]) for s in sorted({r['subject'] for r in records})}
    write(RUN / 'evidence/results.json', dict(rows=rows))
    write(RUN / 'evidence/summary.json', dict(common_pairs=len(paired), overall=summary, subjects=subjects, all_feasible=all_available))
    write(RUN / 'evidence/execution.json', dict(base='204193228c13924179b8176bde9c4522322b7d62',
          commands=['<BODY_PYTHON> -Bu run_fit.py','<BODY_PYTHON> -Bu measure.py'], fit_exit=done['exit_status'],
          eval_exit=0, steps=done['steps'], elapsed_fit_seconds=done['elapsed_seconds'],
          params_sha256=sha(OUT / 'budget_20000.npz'), reference_sha256=source['reference_local_sha256'],
          manifest_sha256=source['manifest_sha256'], evaluation_pair_counts=dict(Counter(x['subject'] for x in refs.values()))))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
