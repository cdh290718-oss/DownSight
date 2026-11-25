"""Resolve the designated research implementation and replay it in a separate output directory."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/golden_d_v4_20261125/check'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('start',type=int)
    parser.add_argument('stride',type=int)
    args = parser.parse_args()
    if not 0 <= args.start < args.stride:
        parser.error('require 0 <= start < stride')
    manifest = json.loads((ROOT/'configs/golden_d.current.json').read_text())
    assert manifest['method_id']=='golden-d-2d-v4'
    entry = ROOT/manifest['implementation']
    sys.path.insert(0,str(entry.parent))
    import lock
    assert Path(lock.__file__).resolve()==entry
    prior = lock.OUT
    lock.OUT = OUT
    (OUT/'poses').mkdir(parents=True,exist_ok=True)
    lock.run(args.start,args.stride)
    sample = lock.prev.read(ROOT/'runs/2026-11-23_fixed_floor24/evidence/sample.json')['rows']
    for row in sample[args.start::args.stride]:
        key = row['pair']
        with np.load(OUT/'poses'/(key+'.npz')) as actual, np.load(prior/'poses'/(key+'.npz')) as expected:
            assert set(actual.files)==set(expected.files)
            assert all(np.array_equal(actual[k],expected[k]) for k in actual.files)
        assert lock.prev.read(OUT/'poses'/(key+'.json'))['selected']==lock.prev.read(prior/'poses'/(key+'.json'))['selected']
        print('exact',key,flush=True)


if __name__=='__main__':
    main()
