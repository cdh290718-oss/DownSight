"""Repeat each locked fit and compare the selected output with the first run."""
import sys
import numpy as np
import lock

first,step=map(int,sys.argv[1:3])
original=lock.OUT
lock.OUT=original/'replay'
(lock.OUT/'poses').mkdir(parents=True,exist_ok=True)
lock.run(first,step)
rows=lock.prev.read(lock.ROOT/'runs/2026-11-23_fixed_floor24/evidence/sample.json')['rows'][first::step]
for row in rows:
    f=row['pair']+'.npz'
    with np.load(original/'poses'/f) as a, np.load(lock.OUT/'poses'/f) as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
print('replayed',len(rows))
