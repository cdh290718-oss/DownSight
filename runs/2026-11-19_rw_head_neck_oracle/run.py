"""Run from the repository root using the existing body environment and PYTHONPATH=."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / 'runs/2026-11-17_newdetector_d10'
OUT = ROOT / 'outputs/rw_head_neck_oracle_20261119'
OLD = ROOT / 'outputs/unrealego_gt2d_ideal_20261115'
sys.path.insert(0, str(REF))
import common

common.RUN = HERE
common.EVIDENCE = HERE / 'evidence'
common.OUT = OUT
read, write, sha = common.read, common.write, common.sha


def prepare():
    assert not (OUT / 'prepared.json').exists()
    reference = read(ROOT / 'configs/golden_d.current.json')
    hashes = {}
    for entry in reference['locked_files']:
        digest = sha(ROOT / entry['path'])
        assert digest == entry['sha256']
        hashes[entry['path']] = digest
    frozen = ROOT / 'outputs/newdetector_d10_20261117/prepared.json'
    items = read(frozen)['items'][:4]
    cfg = read(REF / 'protocol.json')
    cfg['pairs'] = ['body%02d' % n for n in range(1, 5)]
    assert [item['pair'] for item in items] == cfg['pairs']
    cfg.update(scope='Four S9 official bilateral 2D annotation pairs; baseline solver; diagnostic.',
               intervention='Replace the 12 reference detector XY in both views with official XY.')
    for unused in ['fixed_mixed_route', 'statistics', 'secondary']:
        cfg.pop(unused, None)
    cfg['baseline_commit'] = 'f0f08013c1183a09410e6dcdcd7941d066e2f835'
    rows = []
    for item in items:
        old = OLD / item['pair']
        obs = read(old / 'observations_local.json')
        assert item['anchor'] == read(old / 'anchors/oracle.json')
        assert item['inputs'] == read(old / 'inputs.json')
        assert item['condition'] == 'baseline'
        assert [p['name'] for p in item['points']] == cfg['observation_joints']
        for p in item['points']:
            idx = obs['names'].index(p['name'])
            xy = np.array(obs['views'])[:, idx]
            assert obs['indices'][idx] == p['index']
            assert p['valid'] == [True, True] and np.all(xy[:, 2] > 0)
            assert np.isfinite(xy).all() and np.all((xy[:, 0] >= 280) & (xy[:, 0] < 1000))
            assert np.all((xy[:, 1] >= 0) & (xy[:, 1] < 720))
            p['xy'] = xy[:, :2].tolist()
        rows.append(dict(pair=item['pair'], joint_views=2*len(item['points']),
                         anchor_sha256=sha(old/'anchors/oracle.json'),
                         transforms_sha256=sha(old/'inputs.json'),
                         annotations_sha256=sha(old/'observations_local.json')))
    write(OUT/'protocol.json', cfg)
    write(OUT/'prepared.json', dict(items=items))
    write(common.EVIDENCE/'preparation.json', dict(pairs=rows, reference_files=hashes,
          reference_prepared_sha256=sha(frozen), prepared_sha256=sha(OUT/'prepared.json'),
          protocol_sha256=sha(OUT/'protocol.json'), camera_sha256=[sha(f) for f in common.camera_files()]))


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'prepare':
        prepare()
    elif stage == 'fit':
        import fit
        fit.RUN = OUT  # only protocol location changes
        sys.argv = [__file__, '--threads', '1']
        fit.main()
    elif stage == 'evaluate':
        import evaluate
        evaluate.RUN = OUT
        evaluate.main()
        result_file = common.EVIDENCE / 'results.json'
        result = read(result_file)
        result['command'] = '<BODY_PYTHON> ' + str(HERE.relative_to(ROOT) / 'run.py') + ' evaluate'
        write(result_file, result)
    else:
        raise ValueError(stage)
