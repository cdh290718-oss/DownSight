"""Compare candidate binocular wrist ranges with ranges from the fixed Golden D fits."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from collections import Counter
import sys

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'runs/2026-11-19_rw_frozen200_free12'))
import measure as ref

spec = spec_from_file_location('stereo_geometry', ROOT / 'runs/2026-11-04_session-hand6-v2/code/prepare.py')
geometry = module_from_spec(spec)
spec.loader.exec_module(geometry)


def summarize(rows):
    result = {'n': len(rows)}
    if rows:
        for name in ['stereo', 'fit']:
            errors = np.array([r[name + '_range_error_mm'] for r in rows])
            result[name + '_range_rmse_mm'] = float(np.sqrt(np.mean(errors**2)))
            result[name + '_range_median_abs_mm'] = float(np.median(abs(errors)))
        result['stereo_closer'] = sum(abs(r['stereo_range_error_mm']) < abs(r['fit_range_error_mm']) for r in rows)
    return result


def main():
    torch.set_num_threads(1)
    previous = ref.read(ROOT / 'runs/2026-11-23_rw20000_pa12/evidence/results.json')
    for path, digest in previous['input_sha256'].items():
        assert ref.sha(ROOT / path) == digest
    model, camera = ref.common.setup_model()
    cameras = [camera(p) for p in ref.camera_files()]
    items = ref.read(ref.OUT / 'prepared.json')['items']
    ids = ref.read(ref.OUT / 'fit_complete.json')['ids']
    params = dict(np.load(ref.OUT / 'budget_20000.npz'))
    targets = ref.read(ref.SOURCE / 'reference.json')
    rows = []
    for i, (pair, detector, method) in enumerate(ids):
        if detector != 'rw20000':
            continue
        item = items[i]
        assert method == '2d' and item['pair'] == pair and item['condition'] == detector
        transforms = [np.array(item['inputs']['views'][s]['camera']['left_to_view']) for s in ('left', 'right')]
        assert np.allclose(transforms[0], np.eye(4))
        right = transforms[1]
        fit = ref.full_joints(params, model.layer['neutral'], i)
        for point in item['points']:
            if point['group'] != 'wrist':
                continue
            row = {'pair': pair, 'joint': point['name'], 'reason': 'missing view'}
            rows.append(row)
            if not all(point['valid']):
                continue
            rec = geometry.triangulate(cameras, right[:3, :3], right[:3, 3], point['xy'])
            row['reason'] = rec['reason']
            if not rec['accepted']:
                continue
            xyz = np.array(rec['point_cam3_m'])
            unit = xyz / np.linalg.norm(xyz)
            sigma = np.sqrt(unit @ np.array(rec['covariance_cam3_m2']) @ unit)
            gt_index = ref.common.NAMES.index(point['name'])
            assert ref.common.NATIVE_IDS[gt_index] == point['native_index']
            gt_range = np.linalg.norm(targets[pair]['gt'][gt_index])
            row.update(range_sigma_mm=float(1000 * sigma),
                       stereo_range_error_mm=float(1000 * (np.linalg.norm(xyz) - gt_range)),
                       fit_range_error_mm=float(1000 * (np.linalg.norm(fit[point['native_index']]) - gt_range)))
    assert len(rows) == 200
    accepted = [r for r in rows if r['reason'] == 'accepted']
    assert len({(r['pair'], r['joint']) for r in rows}) == len(rows)
    tight = [r for r in accepted if r['range_sigma_mm'] <= 50]
    result = dict(reasons=dict(Counter(r['reason'] for r in rows)), accepted=summarize(accepted),
                  sigma_le_50mm=summarize(tight), rows=rows, input_sha256=previous['input_sha256'],
                  command='<BODY_PYTHON> -Bu runs/2026-11-23_rw20000_wrist_depth/check_depth.py')
    ref.write(RUN / 'evidence/results.json', result)
    print(result['reasons'], result['accepted'], result['sigma_le_50mm'])


if __name__ == '__main__':
    main()
