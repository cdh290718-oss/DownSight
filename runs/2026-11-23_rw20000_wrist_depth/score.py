"""Use the same archived 91 unaffected fits in both arms and score the paired nine."""
import numpy as np
import torch
from fit_range import OUT, RUN, ref
from metrics import align


def summarize(rows):
    def join(name):
        return np.concatenate([np.atleast_1d(r[name]) for r in rows])
    errors = join('joint_mm')
    wrists = join('wrist_mm')
    return dict(n=len(rows), mpjpe_mm=float(errors.mean()), rmse_mm=float(np.sqrt(np.mean(errors**2))),
                pa_mpjpe_mm=float(join('pa_mm').mean()), wrist_mpjpe_mm=float(wrists.mean()),
                wrist_rmse_mm=float(np.sqrt(np.mean(wrists**2))),
                nonwrist_mpjpe_mm=float(join('nonwrist_mm').mean()),
                gt_rmse_px=float(np.sqrt(join('gt_sq_px').sum() / join('n2').sum())))


def main():
    torch.set_num_threads(1)
    old, new = ref.OUT, OUT
    done = ref.read(new / 'fit_complete.json')
    diag = ref.read(new / 'diagnostic_20000.json')
    assert done['exit_status'] == 0 and done['steps'] == 20000 and all(diag['selected_feasible'])
    assert done['protocol_sha256'] == ref.sha(old / 'protocol.json')
    images, ids2 = ref.reference_images()
    truths = ref.read(ref.SOURCE / 'reference.json')
    model, camera = ref.common.setup_model()
    cameras = [camera(p) for p in ref.camera_files()]
    adapters = [ref.read(p)['input_adapter'] for p in ref.camera_files()]
    records = {}
    for source in [old, new]:
        params = dict(np.load(source / 'budget_20000.npz'))
        items = ref.read(source / 'prepared.json')['items']
        completion = ref.read(source / 'fit_complete.json')
        for i, (pair, condition, method) in enumerate(completion['ids']):
            if source == old and condition != 'rw20000':
                continue
            item = items[i]
            assert item['pair'] == pair and method == '2d'
            pred = ref.full_joints(params, model.layer['neutral'], i)[ref.common.NATIVE_IDS][ref.common.OBS]
            gt = np.array(truths[pair]['gt'])[ref.common.OBS]
            native = 1000 * np.linalg.norm(pred-gt, axis=1)
            pa = 1000 * np.linalg.norm(align(pred, gt)-gt, axis=1)
            wrist = np.array([p['group'] == 'wrist' for p in item['points']])
            views = ref.pixel_errors(item, pred, images[pair], ids2, cameras, adapters)
            records.setdefault(pair, {})[condition] = dict(
                joint_mm=native.tolist(), pa_mm=pa.tolist(), wrist_mm=native[wrist].tolist(),
                nonwrist_mm=native[~wrist].tolist(), n2=sum(v['n'] for v in views),
                gt_sq_px=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views))
    triggered = [p for p, v in records.items() if 'range' in v]
    assert len(records) == 100 and len(triggered) == 9
    archive_summary = summarize([v['rw20000'] for v in records.values()])
    assert abs(archive_summary['mpjpe_mm'] - 61.23) < .01
    assert abs(archive_summary['rmse_mm'] - 79.62) < .01
    assert abs(archive_summary['gt_rmse_px'] - 18.69) < .01
    scopes = dict(all=list(records), triggered=triggered,
                  triggered_without_003_004=[p for p in triggered if p not in ['pair003', 'pair004']])
    summary = {scope:{c:summarize([records[p].get(c, records[p]['rw20000']) for p in pairs])
                      for c in ['baseline','range']} for scope,pairs in scopes.items()}
    paired = {p:{c:summarize([records[p][c]]) for c in ['rw20000','baseline','range']} for p in triggered}
    converged = {c:sum(ok for (_, label, _), ok in zip(diag['ids'], diag['convergence_pass']) if label == c)
                 for c in ['baseline', 'range']}
    ref.write(RUN / 'evidence/fit_results.json',dict(summary=summary, archived_baseline=archive_summary, paired=paired, converged=converged,
              feasible=len(done['ids']), steps=done['steps'], status=done['exit_status'],
              fit_seconds=done['elapsed_seconds'], params_sha256=ref.sha(new/'budget_20000.npz'),
              settings=dict(sigma_max_mm=50, range_coefficient=.25, global_metric_weight=.01,
                            huber_delta=2.5, denominator=2), input_protocol_sha256=done['protocol_sha256']))
    print(summary, converged)


if __name__ == '__main__':
    main()
