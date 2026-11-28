"""Fit the same body objective with two known, frame-matched Head positions."""
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/rw12_body_head_anchors_20261128'
CONTROL = ROOT / 'outputs/rw12_head_contour_only_20261128'
PREVIOUS = ROOT / 'outputs/rw12_headonly_20261128'
SOURCE = ROOT / 'runs/2026-11-28_rw12_head_contour_only'

sys.path.insert(0, str(SOURCE))
from body_only import BODY, TorsoOnly, camera_paths, read, sha, v3, write

ANCHORS = ('mount_nose', 'body_contour')


class ExactHead(TorsoOnly):
    def forward(self):
        joints, rotation, betas = super().forward()
        offset = self.head_pos - joints[:, 15]
        return joints + offset.unsqueeze(1), rotation, betas


def prepare():
    path = CONTROL / 'prepared.json'
    old = read(SOURCE / 'evidence/inputs.json')
    assert sha(path) == old['body_inputs_sha256']
    assert sha(CONTROL / 'protocol.json') == old['protocol_sha256']
    rows = read(path)['items']
    assert len(rows) == 12
    prepared, receipts = [], {}
    for row in rows:
        pair = row['pair']
        paths = [PREVIOUS / 'poses' / f'{pair}_head.json', CONTROL / 'poses' / f'{pair}.json']
        for name, source in zip(ANCHORS, paths):
            data = read(source)
            assert data['pair'] == pair and data['exit_status'] == 0
            xyz = data['final_head_m']
            assert np.asarray(xyz).shape == (3,) and np.isfinite(xyz).all()
            prepared.append(dict(row, condition=name, head_position=xyz))
            receipts.setdefault(pair, {})[name] = dict(position_m=xyz, source_sha256=sha(source))
    cfg = read(CONTROL / 'protocol.json')
    # The fitter uses the baseline condition to build its fixed mean-shape basis.
    for row in prepared:
        if row['condition'] == ANCHORS[0]:
            row['condition'] = 'baseline'
    cfg.update(conditions=['baseline', ANCHORS[1]],
               scope='Matched 12-point body fit with exact per-frame Head position')
    assert cfg['head_position_weight'] == cfg['head_rotation_weight'] == 0
    write(OUT / 'prepared.json', dict(items=prepared))
    write(OUT / 'protocol.json', cfg)
    write(RUN / 'evidence/inputs.json', dict(
        observations_sha256=sha(path), prepared_sha256=sha(OUT / 'prepared.json'),
        protocol_sha256=sha(OUT / 'protocol.json'), previous_control_protocol_sha256=old['protocol_sha256'],
        anchor_type='exact', head_rotation_weight=0, rows=receipts, exit_status=0))


def seal():
    """Export actual root translations from the projected Head constraint."""
    fit = read(OUT / 'fit_complete.json')
    assert len(fit['ids']) == 24 and fit['steps'] == 20000 and fit['exit_status'] == 0
    prepared = read(OUT / 'prepared.json')['items']
    model, camera = v3.ref.common.setup_model()
    problem = ExactHead(prepared, read(OUT / 'protocol.json'), model, camera)
    source = OUT / 'budget_20000.npz'
    with np.load(source) as archive:
        pack = dict(archive)
    assert np.all(pack['selected_step'] >= 0)
    with torch.no_grad():
        tensors = [problem.t(pack[field]) for field in ('body_pose', 'betas', 'global0', 'delta')]
        raw, _ = problem.fk(*tensors, torch.zeros_like(problem.tr))
        pack['translation'] = (problem.head_pos - raw[:, 15]).numpy()
        joints, _ = problem.fk(*tensors, problem.t(pack['translation']))
        residual = (joints[:, 15] - problem.head_pos).norm(dim=1)
        assert residual.max() < 1e-6
    destination = OUT / 'anchored_body.npz'
    with destination.open('xb') as stream:
        np.savez_compressed(stream, **pack)
    write(OUT / 'sealed.json', dict(sha256=sha(destination), optimizer_checkpoint_sha256=sha(source),
                                   fit_complete_sha256=sha(OUT / 'fit_complete.json'),
                                   max_fk_anchor_error_mm=float(residual.max()) * 1000, exit_status=0))


if __name__ == '__main__':
    torch.set_num_threads(1)
    BODY.camera_files = camera_paths
    task = sys.argv[1]
    if task == 'prepare':
        prepare()
    elif task == 'fit':
        BODY.RUN = BODY.OUT = OUT
        BODY.Problem = ExactHead
        sys.argv = [__file__, '--threads', '1']
        BODY.main()
    else:
        assert task == 'seal'
        seal()
