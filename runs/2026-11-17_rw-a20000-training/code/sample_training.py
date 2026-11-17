"""Nested, deterministic paired sampling from the official training partition."""
import collections
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, d): p.write_text(json.dumps(d, indent=2) + '\n')

def main():
    source = ROOT.parent / 'EgoSMPLX/data/UnrealEgo_RW_Train'
    prior = read(ROOT / 'runs/2026-11-17_rw-a-scale-prep/evidence/inventory.json')
    assert sha(source / 'prepared/samples.npz') == prior['samples_sha256']
    assert sha(source / 'official_splits/UnrealEgoData_rw.zip') == prior['split_sha256']
    with zipfile.ZipFile(source / 'official_splits/UnrealEgoData_rw.zip') as z:
        official = set(z.read('UnrealEgoData_rw/train.txt').decode().splitlines())
    holdout = prior['preserved_old_same_person_holdout']
    with np.load(source / 'prepared/samples.npz', allow_pickle=False) as z:
        seq, frame, side, paths = [z[k] for k in ['source_sequence','frame_index','camera_side','image_path']]
    pairs = collections.defaultdict(dict)
    for i, s in enumerate(seq):
        s = str(s)
        if s in official and s != holdout:
            assert str(side[i]) not in pairs[(s, int(frame[i]))]
            pairs[(s, int(frame[i]))][str(side[i])] = i
    frames = collections.defaultdict(list)
    for (s, f), views in sorted(pairs.items()):
        if set(views) == {'left', 'right'} :
            frames[s].append(f)
    oldpath = ROOT / 'outputs/rw_a_scale_prep_20261117_v2/train_selection.json'
    old = read(oldpath)['records']
    assert len(old) == 5000
    selected = {s: set() for s in frames}
    for row in old:
        assert row['frame'] in frames[row['sequence']]
        selected[row['sequence']].add(row['frame'])
    # Retain all old anchors; build a spaced candidate pool around them.
    for s, available in frames.items():
        kept = set(selected[s])
        for f in available:
            if all(abs(f-g) >= 5 for g in kept): kept.add(f)
        frames[s] = sorted(kept)
    subjects = sorted({s.split('/')[2] for s in frames})
    assert len(subjects) == 9
    capacity = {u: sum(len(v) for s,v in frames.items() if s.split('/')[2] == u) for u in subjects}
    assert sum(capacity.values()) >= 10000
    quota = dict.fromkeys(subjects, 0)
    while sum(quota.values()) < 10000:
        for u in subjects:
            if quota[u] < capacity[u] and sum(quota.values()) < 10000: quota[u] += 1
    for u in subjects:
        sequences = sorted(s for s in frames if s.split('/')[2] == u)
        count = sum(len(selected[s]) for s in sequences)
        while count < quota[u]:
            s = min((s for s in sequences if len(selected[s]) < len(frames[s])), key=lambda s: (len(selected[s]), s))
            candidates = set(frames[s]) - selected[s]
            f = max(candidates, key=lambda f: (min(abs(f-g) for g in selected[s]) if selected[s] else len(frames[s]), -f))
            selected[s].add(f); count += 1
    records = []
    for s in sorted(selected):
        for f in sorted(selected[s]):
            for eye in ['left','right']:
                i = pairs[(s,f)][eye]
                records.append(dict(source_index=i, sequence=s, subject=s.split('/')[2], frame=f,
                                    view=eye, pair_id=s+'/'+str(f), original_cache_key=str(paths[i])))
    assert len(records) == 20000 and {r['source_index'] for r in old} <= {r['source_index'] for r in records}
    assert len({r['source_index'] for r in records}) == 20000
    gaps = [b-a for vv in selected.values() for a,b in zip(sorted(vv),sorted(vv)[1:])]
    assert min(gaps) >= 5
    out = ROOT / 'outputs/rw_a20000_selection_20261117'; out.mkdir(exist_ok=False)
    write(out/'train_selection.json',dict(records=records))
    cfg = read(RUN/'config.json');cfg['training_selection_sha256'] = sha(out/'train_selection.json');write(RUN/'config.json',cfg)
    write(RUN/'evidence/selection.json',dict(command='python code/sample_training.py',exit_status=0,
        images=len(records),pairs=len(records)//2,subjects=subjects,subject_pair_quota=quota,
        eligible_spaced_pair_capacity=capacity,sequences=len(selected),min_source_frame_gap=min(gaps),
        original_5000_all_retained=True,original_selection_sha256=sha(oldpath),
        selection_sha256=sha(out/'train_selection.json'),excluded_holdout_sequence=holdout,
        official_train_only=True,sampling='Nested subject/sequence balancing; farthest source-frame index within sequence; minimum index gap 5, not a time interval.',
        limitation='Cache presence audited previously; every selected image must still be decoded during preparation.'))
    print(json.dumps(dict(images=len(records),subject_pairs=quota,capacity=capacity,min_gap=min(gaps))))

if __name__ == '__main__': main()
