"""Check 2D against archived per-image errors and 3D on archived frozen fits."""
import numpy as np

from measure import (GOLD, MODELS, OUT, ROOT, RUN, SOURCE, camera_files, common, full_joints,
                     joint_errors, pixel_errors, read, reference_images, sha, write)


def main():
    model, camera = common.setup_model()
    cams = [camera(p) for p in camera_files()]
    adapters = [read(p)['input_adapter'] for p in camera_files()]
    source = read(GOLD / 'evidence/preparation.json')
    assert all(sha(p)==c['local_sha256'] for p,c in zip(camera_files(),source['cameras']))
    images, joint_ids = reference_images()
    ref = read(SOURCE / 'reference.json')
    old_rows = read(GOLD / 'evidence/results.json')['records']
    actual = read(OUT / 'prepared.json')['items']
    pixel_ref = {}
    for run in ['2026-11-18_rw-frozen200-evaluation','2026-11-18_rw20000_frozen200']:
        for row in read(ROOT / 'runs' / run / 'evidence/2d_errors.json')['records']:
            key = (row['version'], row['id'])
            assert key not in pixel_ref
            pixel_ref[key] = [e for e in row['per_joint_error_px'] if e is not None]
    assert len(pixel_ref)==800
    max_sse_delta, max_mpjpe_delta = 0., 0.
    with np.load(SOURCE / 'budget_20000.npz') as data:
        old_params = dict(data)
    for i,row in enumerate(actual):
        m = 'original' if row['condition']=='baseline' else row['condition']
        pair = row['pair']
        views = pixel_errors(row,np.asarray(ref[pair]['gt'])[common.OBS],images[pair],joint_ids,cams,adapters)
        for v,view in enumerate(views):
            errors = pixel_ref[m,images[pair][v]['id']]
            assert len(errors)==view['n']
            diff = abs(np.square(errors).sum()-view['sum_sq_px2']['detection_vs_gt'])
            max_sse_delta = max(diff, max_sse_delta)
        if pair in ('pair001','pair050','pair100'):
            native = full_joints(old_params, model.layer['neutral'].eval(), i)[common.NATIVE_IDS]
            gt = np.asarray(ref[pair]['gt'])
            values = joint_errors(native[common.OBS], gt[common.OBS]) + joint_errors(native, gt)
            old = old_rows[i]
            assert (old['pair'],old['model'])==(pair,m)
            expected = [old[k] for k in ('absolute_12_mpjpe_mm','pa_12_mpjpe_mm','absolute_mpjpe_mm','pa_mpjpe_mm')]
            max_mpjpe_delta = max(max_mpjpe_delta, max(abs(np.array(values)-expected)))
    assert max_sse_delta < .001 and max_mpjpe_delta < .01, (max_sse_delta,max_mpjpe_delta)
    out = dict(views_checked=len(pixel_ref), max_detector_sse_delta_px2=max_sse_delta,
               archived_fits_checked=12, max_archived_mpjpe_delta_mm=max_mpjpe_delta,
               detector_all_200_px={})
    for m in MODELS:
        e = [v for (name,_),vs in pixel_ref.items() if name==m for v in vs]
        assert len(e)==2398
        out['detector_all_200_px'][m]=dict(points=len(e), radial_rmse=float(np.sqrt(np.mean(np.square(e)))))
    write(RUN / 'evidence/verification.json', out)
    print(out)


if __name__ == '__main__':
    main()
