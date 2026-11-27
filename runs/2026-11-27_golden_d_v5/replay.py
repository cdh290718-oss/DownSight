"""Replay the frozen hand stage from the four reference body seeds in an isolated output."""
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/golden_d_v5_20261127'

def read(path):
    return json.loads(path.read_bytes())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = read(ROOT/'configs/golden_d.2d-v5.json')
    assert manifest['method_id']=='golden-d-2d-v5'
    body_file = ROOT/manifest['previous_manifest']
    assert digest(body_file)==manifest['previous_manifest_sha256']
    for version in (manifest, read(body_file)):
        for item in version['locked_files']:
            assert digest(ROOT/item['path'])==item['sha256'], item['path']
    entry = ROOT/manifest['implementation']
    spec = importlib.util.spec_from_file_location('v5_hand_replay',entry)
    method = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)
    method.OUT = OUT
    method.run()
    actual = read(OUT/'fit.json')['frames']
    expected = read(ROOT/manifest['reference_results'])['frames']
    assert set(actual)==set(expected)==set(manifest['reference_cases'])
    checks = {}
    for key, row in actual.items():
        ref = expected[key]
        assert row['seed_sha256']==ref['seed_sha256']
        assert row['hands']==ref['hands']
        obj_sha = digest(OUT/'export'/(key+'_wrist_cam3_m.obj'))
        assert obj_sha==row['obj_sha256']==ref['obj_sha256']
        with np.load(OUT/(key+'.npz')) as result, np.load(method.prior.OUT/'poses'/(key+'_wrist.npz')) as previous:
            assert set(result.files)==set(previous.files)
            assert all(np.array_equal(result[k],previous[k]) for k in result.files)
        checks[key] = dict(obj_sha256=obj_sha,seed_sha256=row['seed_sha256'],parameters_exact=True,
                           hand_sources={s:h['source'] for s,h in row['hands'].items()})
    (OUT/'verification.json').write_text(json.dumps(dict(method=manifest['method_id'],frames=checks,
        frozen_manifest_sha256=digest(ROOT/'configs/golden_d.2d-v5.json'),exit_status=0),indent=2)+'\n')
    print('Verified',len(checks),'body parameter packs and full-body meshes')

if __name__=='__main__':
    main()
