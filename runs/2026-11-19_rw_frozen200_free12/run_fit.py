"""Repeat the position-only head experiment on the frozen four-detector inputs."""
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'runs/2026-11-19_rw_three_free12'))
from experiment import common, fit, read, sha, shoulder_hip_recovery, trial, write

OUT = ROOT / 'outputs/rw_frozen200_free12_20261119'
SOURCE = ROOT / 'outputs/rw_frozen200_golden_d_20261118'
GOLD = ROOT / 'runs/2026-11-18_rw_frozen200_golden_d'
MODELS = ['original', 'rw500', 'rw5000', 'rw20000']


def camera_files():
    return [SOURCE / ('camera_' + side + '.json') for side in ('left', 'right')]


def main():
    provenance = read(GOLD / 'evidence/preparation.json')
    assert sha(SOURCE / 'prepared.json') == provenance['prepared_local_sha256']
    rows = read(SOURCE / 'prepared.json')['items']
    cfg = read(GOLD / 'protocol.json')
    cfg.update(head_rotation_weight=0., landmark_ratios=read(trial.PREV / 'evidence/calibration.json')['ratios'],
               scope='Frozen 100 pairs, four detectors, position-only head and free feet.')
    model, camera = common.setup_model()
    layer = model.layer['neutral']
    rest = (layer.J_regressor @ layer.v_template).detach().numpy()
    cams = [camera(p) for p in camera_files()]
    prepared, audit, shared = [], [], {}
    for row in rows:
        pair, model_name = row['base_pair'], row['detector']
        condition = 'baseline' if model_name == 'original' else model_name
        head = row['anchor']['position']
        points = [{k: p[k] for k in ('name', 'index', 'native_index', 'group', 'xy', 'valid', 'accepted')}
                  for p in row['points']]
        assert len(points) == 12 and not any(p['accepted'] for p in points)
        nuisance = [head, row['inputs'], [p['valid'] for p in points]]
        assert shared.setdefault(pair, nuisance) == nuisance
        seed, reason = trial.body_seed(row, cams)
        if reason:
            seed = shoulder_hip_recovery(row, cams, rest)
        assert np.allclose(seed.T @ seed, np.eye(3)) and np.isclose(np.linalg.det(seed), 1)
        prepared.append(dict(pair=pair, condition=condition, head_position=head, points=points,
                             inputs=row['inputs'], initial_body_rotation=seed.tolist()))
        audit.append(dict(pair=pair, model=model_name, subject=row['subject'], stereo_seed_issue=reason,
                          view_points=sum(sum(p['valid']) for p in points)))
    assert len(prepared) == 400 and len(shared) == 100
    assert len({(x['pair'],x['model']) for x in audit}) == 400
    for name in MODELS:
        assert sum(x['view_points'] for x in audit if x['model'] == name) == 2398
    write(OUT / 'prepared.json', dict(items=prepared))
    write(OUT / 'protocol.json', cfg)
    write(RUN / 'evidence/inputs.json', dict(rows=audit, source_sha256=sha(SOURCE / 'prepared.json'),
          solver_input_sha256=sha(OUT / 'prepared.json'), protocol_sha256=sha(OUT / 'protocol.json'),
          cameras={p.name:sha(p) for p in camera_files()}, prediction_sha256=provenance['prediction_sha256']))
    def protect_reference(event, args):
        if event == 'open' and args and isinstance(args[0], str) and Path(args[0]).name == 'reference.json':
            raise PermissionError('Evaluation reference is unavailable during fitting.')
    sys.addaudithook(protect_reference)
    fit.RUN = fit.OUT = OUT
    fit.Problem = trial.TorsoOnly
    fit.camera_files = camera_files
    sys.argv = [__file__, '--threads', '1']
    fit.main()


if __name__ == '__main__':
    main()
