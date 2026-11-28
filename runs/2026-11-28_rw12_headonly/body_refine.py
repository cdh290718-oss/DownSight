"""Apply the Video 003 body stage on the new mount-based RW fits."""
import sys

import cv2
import numpy as np
import torch

from body_start import ROOT, RUN, OUT, SIDES, camera_paths, read, sha, write
sys.path.insert(0, str(ROOT / 'runs/2026-11-25_floor24_foot_lock'))
from lock import FixedFeet, WEIGHT, compare, from_branch, prev, search
import candidates as wrist

SEG = ROOT / 'outputs/frozen100_outline_20261125/seg'


def main(index):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    compare.OUT = SEG.parent
    prev.ref.camera_files = prev.base.fit.camera_files = camera_paths
    folder = OUT / 'fit/0'
    inputs = read(RUN / 'evidence/inputs.json')
    assert sha(folder / 'prepared.json') == inputs['shards']['0']
    assert sha(folder / 'protocol.json') == inputs['protocol']
    cfg = read(folder / 'protocol.json')
    row = read(folder / 'prepared.json')['items'][index]
    key = row['pair']
    records = read(folder / 'selected.json')[index]
    assert records['pair'] == key
    with np.load(folder / 'budget_20000.npz') as bank:
        raw = dict(bank)
    with np.load(folder / 'selected.npz') as bank:
        orig = {field: bank[field][records['offset'] + 2] for field in search.FIELDS}
    base = {field: raw[field][index] for field in search.FIELDS}
    seed = {field: value.copy() for field, value in base.items()}
    for side in SIDES:
        assert sha(SEG / f'{key}_{side}.npy') == inputs['rows'][key]['eyes'][side]['probability_sha256']
    photos = [cv2.imread(str(OUT / 'images' / f'{key}_{side}.png')) for side in SIDES]
    model, camera = prev.ref.common.setup_model()
    with torch.no_grad():
        arms = wrist.frame_candidates(row, raw, index, model, camera, photos,
                                     [read(p)['input_adapter'] for p in camera_paths()], cfg)
        for (shoulder, elbow, _), options in zip(wrist.ARMS, arms):
            seed['body_pose'][[shoulder - 1, elbow - 1]] = min(options, key=lambda x: x['loss'])['pose']
        branches, probes = search.frame(row, seed, base, model, camera, cfg)
        best = min(branches, key=lambda x: x['loss'])
        assert records['probes'] == probes and records['selected'][-1] == best['name']
        assert all(np.array_equal(best[k], orig[k]) for k in search.FIELDS[:3])
    frame = FixedFeet(key, row, orig, model, camera, cfg)
    evolved, _ = frame.descend(frame.v3, False, True, 'v3')
    fitted = dict(orig, body_pose=evolved[:21].numpy(), delta=evolved[21].numpy(),
                  translation=evolved[22].numpy())
    with torch.no_grad():
        new, _ = search.frame(row, fitted, fitted, model, camera, cfg)
        pool = {'v3': frame.v3, 'adam': evolved}
        pool.update({'v3_' + x['name']: from_branch(frame.v3, x) for x in branches})
        pool.update({'new_' + x['name']: from_branch(evolved, x) for x in new})
        scores = {}
        for name, pose in pool.items():
            vertices, _, loss, valid = frame.state(pose)
            if valid:
                scores[name] = (loss + WEIGHT * frame.observed(vertices)[0]).item()
    winner = min(scores, key=scores.get)
    path = OUT / 'poses' / f'{key}_body.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, locked=pool[winner].numpy(), global0=orig['global0'], betas=orig['betas'])
    write(path.with_suffix('.json'), dict(pair=key, source='mount-based DV4', finalist=winner,
          sha256=sha(path), exit_status=0))
    print(key, winner, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
