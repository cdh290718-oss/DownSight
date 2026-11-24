"""Paired 12-joint evaluation and strict triangle crossings on diagnostic frames."""
from collections import Counter
import sys

import numpy as np
import torch

from select_pose import EV, OUT, ROOT, prev, search, vertices
from contact import TOL_MM, groups, preserves_near
from inspect_obj import self_intersections


def count_hits(verts, faces, labels):
    hits = self_intersections(verts, faces)
    return dict(Counter(' / '.join(sorted([labels[a], labels[b]])) for a, b in hits))


def measure(split):
    torch.set_num_threads(1)
    model, camera = prev.ref.common.setup_model()
    layer = model.layer['neutral']
    faces = np.asarray(model.face)
    labels = groups(layer, faces)
    cameras = [camera(p) for p in prev.ref.camera_files()]
    adapters = [prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    ids2 = prev.mapping()
    obs = prev.ref.common.OBS
    natives = np.asarray(prev.ref.common.NATIVE_IDS)[obs]
    selection = prev.read(EV/(split+'_selection.json'))
    path = OUT/(split+'_selected.npz')
    assert prev.sha(path) == selection['parameter_sha256']
    selected = dict(np.load(path))
    assert not selected['betas'].any()
    archived = dict(np.load(search.OUT/(split+'_bank.npz')))
    choices = prev.read(search.OUT/(split+'_choices.json'))
    baseline = prev.read(search.RUN/('evidence/'+split+'_results.json'))
    manifest, folder, truthfile = ((prev.TM,prev.base.OUT,prev.ref.SOURCE/'reference.json') if split=='test'
                                  else (prev.VM,prev.VAL,prev.OUT/'validation_truth.json'))
    assert prev.sha(truthfile) == baseline['truth_sha256'] and prev.sha(manifest) == baseline['manifest_sha256']
    truth = prev.read(truthfile)
    images = prev.pair_images(manifest)
    items = prev.read(folder/'prepared.json')['items']
    old_diag = prev.read(ROOT/'runs/2026-11-23_floor24_intersections/evidence/results.json')['rows']
    rows = {}
    for i, pair in enumerate(selected['pairs'].tolist()):
        record = choices[pair]
        j = record['offset'] + record['best']
        item = items[record['index']]
        im = images[int(pair[-3:])-1]
        assert pair == item['pair'] and im[0]['subject'] == truth[pair]['subject']
        target = np.asarray(truth[pair]['gt'])[obs]
        s = selection['rows'][pair]
        assert preserves_near(s['before'],s['after'])
        row = dict(subject=im[0]['subject'], selected=s['selected'])
        for key, pack, idx in [('before', archived, j), ('after', selected, i)]:
            pred = prev.ref.full_joints(pack, layer, idx)[natives]
            views = prev.ref.pixel_errors(item,pred,im,ids2,cameras,adapters)
            geom = s[key].values()
            result = dict(nonpa=float(1000*np.linalg.norm(pred-target,axis=1).mean()),
                          pa=float(1000*np.linalg.norm(prev.align(pred,target)-target,axis=1).mean()),
                          n_px=sum(v['n'] for v in views),
                          px_sq=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views),
                          max_depth_mm=max(d for d,g in geom))
            if pair in old_diag:
                params = {k:pack[k][idx] for k in search.FIELDS}
                result['crossings'] = count_hits(vertices(layer,params),faces,labels)
            row[key] = result
        if pair in old_diag:
            row['historical_fixed_crossings'] = old_diag[pair]['surface_crossing_face_pairs']
        old = baseline['rows'][pair]['scores']['joint']
        assert abs(row['before']['pa']-old['pa']) < 1e-4 and abs(row['before']['nonpa']-old['mpjpe']) < 1e-4
        rows[pair] = row
    summary = {}
    for key in ['before','after']:
        rr = [r[key] for r in rows.values()]
        ds = [r['max_depth_mm'] for r in rr]
        summary[key] = dict(pa_mm=np.mean([r['pa'] for r in rr]), nonpa_mm=np.mean([r['nonpa'] for r in rr]),
                            gt_px_rmse=float(np.sqrt(sum(r['px_sq'] for r in rr)/sum(r['n_px'] for r in rr))),
                            flagged=int(sum(d>TOL_MM for d in ds)), mean_frame_max_depth_mm=np.mean(ds), max_depth_mm=max(ds))
    summary['frames'] = len(rows)
    summary['changed'] = sum(r['selected']!='v3' for r in rows.values())
    prev.write(EV/(split+'_evaluation.json'), dict(summary=summary, rows=rows, exit_status=0,
        truth_sha256=prev.sha(truthfile),manifest_sha256=prev.sha(manifest),selection_sha256=prev.sha(path)))
    print(split,summary)


if __name__ == '__main__':
    measure(sys.argv[1])
