"""Initialize shoulder and hip alignment and fit the original body point objective."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/rw12_head_contour_only_20261128'
PRIOR = ROOT / 'outputs/rw12_headonly_20261128'
REFERENCE = ROOT / 'runs/2026-11-28_rw12_headonly'
sys.path.insert(0, str(REFERENCE))
from body_start import v3, camera_paths, read, sha, write

BODY = v3.base.fit
TorsoOnly = v3.base.trial.TorsoOnly
NAMES = ('L_Shoulder', 'R_Shoulder', 'L_Hip', 'R_Hip')


def torso_translation(item, cams, rest):
    transforms = [np.asarray(item['inputs']['views'][s]['camera']['left_to_view'])
                  for s in ('left', 'right')]
    origins = [-t[:3, :3].T @ t[:3, 3] for t in transforms]
    seed = np.asarray(item['initial_body_rotation'])
    offsets = []
    for point in item['points']:
        if point['name'] not in NAMES or not all(point['valid']):
            continue
        left, right = [t[:3, :3].T @ cam.camera2world_ray(np.asarray([xy]))[0]
                       for xy, t, cam in zip(point['xy'], transforms, cams)]
        rays = np.column_stack((left, -right))
        depth, _, rank, _ = np.linalg.lstsq(rays, origins[1] - origins[0], rcond=None)
        if rank != 2 or np.linalg.cond(rays) > 150 or np.min(depth) <= 0:
            continue
        midpoint = (origins[0] + depth[0] * left + origins[1] + depth[1] * right) / 2
        offsets.append(midpoint - (seed @ (rest[point['native_index']] - rest[0]) + rest[0]))
    assert len(offsets) >= 2, item['pair']
    return np.median(offsets, axis=0), len(offsets)


class BodyOnly(TorsoOnly):
    def __init__(self, rows, cfg, model, camera_cls):
        assert cfg['head_position_weight'] == cfg['head_rotation_weight'] == 0
        # The inherited constructor expects this slot. Replace its temporary
        # translation before the optimizer or any forward pass uses it.
        super().__init__([dict(r, head_position=[0., 0., 0.]) for r in rows], cfg, model, camera_cls)
        with torch.no_grad():
            self.tr.copy_(self.t(np.array([r['initial_translation'] for r in rows])))


def prepare():
    prior = REFERENCE / 'evidence/inputs.json'
    path = PRIOR / 'fit/0/prepared.json'
    assert sha(path) == read(prior)['shards']['0']
    rows = read(path)['items']
    cfg = read(PRIOR / 'fit/0/protocol.json')
    cfg.update(head_position_weight=0., head_rotation_weight=0.,
               scope='12 body points only; triangulated shoulder-hip initialization')
    model, camera = v3.ref.common.setup_model()
    layer = model.layer['neutral']
    rest = (layer.J_regressor @ layer.v_template).detach().numpy()
    cams = [camera(p) for p in camera_paths()]
    records = {}
    for row in rows:
        del row['head_position']
        translation, n = torso_translation(row, cams, rest)
        row['initial_translation'] = translation.tolist()
        records[row['pair']] = dict(torso_points=n, initial_translation_m=translation.tolist())
    write(OUT / 'prepared.json', dict(items=rows))
    write(OUT / 'protocol.json', cfg)
    write(RUN / 'evidence/inputs.json', dict(body_inputs_sha256=sha(OUT / 'prepared.json'),
          protocol_sha256=sha(OUT / 'protocol.json'), source_observations_sha256=sha(path),
          source_receipt_sha256=sha(prior), rows=records, head_position_weight=0., head_rotation_weight=0.))


if __name__ == '__main__':
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        BODY.RUN = BODY.OUT = OUT
        BODY.Problem = BodyOnly
        BODY.camera_files = camera_paths
        sys.argv = [__file__, '--threads', '1']
        BODY.main()
