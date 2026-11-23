"""Rescore frozen camera-frame fits using a nested set of framewise alignments."""
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
BASE = ROOT / 'runs/2026-11-19_rw_frozen200_free12'
sys.path.insert(0, str(BASE))
import measure as ref
from metrics import align


def metrics(distances):
    return dict(rmse_mm=float(np.sqrt(np.mean(distances**2))),
                mpjpe_mm=float(distances.mean()))


def main():
    torch.set_num_threads(1)
    previous = ref.read(ROOT / 'runs/2026-11-20_rw20000_joint_rmse/evidence/results.json')
    for path, digest in previous['input_sha256'].items():
        assert ref.sha(ROOT / path) == digest

    params = dict(np.load(ref.OUT / 'budget_20000.npz'))
    ids = ref.read(ref.OUT / 'fit_complete.json')['ids']
    reference = ref.read(ref.SOURCE / 'reference.json')
    archive = {r['pair']: r for r in ref.read(BASE / 'evidence/results.json')['rows']
               if r['model'] == 'rw20000'}
    model, _ = ref.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    names = [ref.common.NAMES[i] for i in ref.common.OBS]
    frames = []
    radial_errors = []
    errors = {key: [] for key in ['camera', 'translation', 'rigid', 'pa']}
    for i, (pair, detector, method) in enumerate(ids):
        if detector != 'rw20000':
            continue
        assert method == '2d'
        target = np.asarray(reference[pair]['gt'])[ref.common.OBS]
        fitted = ref.full_joints(params, layer, i)[ref.common.NATIVE_IDS][ref.common.OBS]
        rays = target / np.linalg.norm(target, axis=1, keepdims=True)
        radial_errors.append(1000 * np.sum((fitted - target) * rays, axis=1))
        x = fitted - fitted.mean(0)
        y = target - target.mean(0)
        u, _, vt = np.linalg.svd(x.T @ y)
        u[:, -1] *= np.sign(np.linalg.det(u @ vt))
        rigid = x @ (u @ vt) + target.mean(0)
        aligned = [fitted, x + target.mean(0), rigid, align(fitted, target)]
        for key, points in zip(errors, aligned):
            errors[key].append(1000 * np.linalg.norm(points - target, axis=1))
        assert np.isclose(errors['camera'][-1].mean(), archive[pair]['mpjpe12_mm'], atol=.001)
        assert np.isclose(errors['pa'][-1].mean(), archive[pair]['pa12_mm'], atol=.001)
        frames.append(pair)

    assert len(set(frames)) == len(frames) == 100
    summary = {}
    for key, rows in errors.items():
        distances = np.array(rows)
        summary[key] = dict(overall=metrics(distances),
                            joints={name: metrics(distances[:, j]) for j, name in enumerate(names)})
    for name in names:
        assert np.isclose(summary['camera']['joints'][name]['rmse_mm'],
                          previous['joints'][name]['fit_rmse_mm'], atol=.001)
    squared_radial = np.square(radial_errors)
    squared_total = np.square(errors['camera'])
    radial_fraction = dict(overall=float(squared_radial.sum() / squared_total.sum()),
                           joints=dict(zip(names, (squared_radial.sum(0) / squared_total.sum(0)).tolist())))
    ref.write(RUN / 'evidence/results.json', dict(pairs=len(frames), joints_per_frame=len(names),
              summary=summary, camera_radial_fraction_of_squared_error=radial_fraction, input_sha256=previous['input_sha256'],
              command='<BODY_PYTHON> -Bu runs/2026-11-23_rw20000_pa12/evaluate_pa.py'))


if __name__ == '__main__':
    main()
