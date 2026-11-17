"""Read-only RW cache audit and deterministic paired sampling; no training."""
import collections
import hashlib
import io
import json
import sqlite3
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
BACKEND = ROOT.parent / 'EgoSMPLX'
DATA = BACKEND / 'data/UnrealEgo_RW_Train'
CACHE = BACKEND / 'data/image_cache/egosmplx_training_safe_20260726'
OUT = ROOT / 'outputs/rw_a_scale_prep_20261117_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def balanced_quota(capacity, total):
    quota = dict.fromkeys(sorted(capacity), 0)
    assert sum(capacity.values()) >= total, 'Insufficient pair capacity'
    while sum(quota.values()) < total:
        for key in quota:
            if quota[key] < capacity[key] and sum(quota.values()) < total:
                quota[key] += 1
    return quota


def main():
    OUT.mkdir(exist_ok=False)
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    assert manifest['format'] == 'egosmplx_image_sqlite_cache_v1'
    assert not (CACHE / '.building').exists()
    with zipfile.ZipFile(DATA / 'official_splits/UnrealEgoData_rw.zip') as z:
        splits = {s: set(z.read('UnrealEgoData_rw/' + s + '.txt').decode().splitlines())
                  for s in ['train', 'validation', 'test']}
    assert not any(splits[a] & splits[b] for a, b in
                   [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    with np.load(DATA / 'prepared/samples.npz', allow_pickle=False) as z:
        seq, frame, side, paths = [z[k] for k in
                                  ['source_sequence', 'frame_index', 'camera_side', 'image_path']]
        xy = z['keypoints2d']
    sequence_split = {s: part for part, ss in splits.items() for s in ss}
    assert all(str(s) in sequence_split for s in seq)
    # The cache's official reader uses MD5 of the absolute original image key.
    shard_names = manifest['shards']
    expected = collections.defaultdict(list)
    for i, path in enumerate(paths):
        key = str(path)
        shard = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(shard_names)
        expected[shard].append((i, key))
    present = np.zeros(len(paths), dtype=bool)
    connections = {}
    shards = []
    for shard, entries in sorted(expected.items()):
        p = CACHE / shard_names[shard]
        c = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True)
        connections[shard] = c
        # Read the key index, never the full image blobs or the full database.
        keys = {r[0] for r in c.execute('SELECT key FROM images')}
        for i, key in entries:
            present[i] = key in keys
        shards.append(dict(name=p.name, bytes=p.stat().st_size,
                           matching_rows=sum(key in keys for _, key in entries)))
    print('Indexed cache keys; matched', int(present.sum()), '/', len(paths), flush=True)
    pairs = collections.defaultdict(dict)
    for i in np.flatnonzero(present):
        key = (str(seq[i]), int(frame[i]))
        assert str(side[i]) not in pairs[key], 'Duplicate view'
        pairs[key][str(side[i])] = int(i)
    complete = {k: v for k, v in pairs.items() if set(v) == {'left', 'right'}}
    counts = {}
    for split in splits:
        subjects = sorted({s.split('/')[2] for s in splits[split]})
        counts[split] = {subject: dict(
            official_sequences=sum(s.split('/')[2] == subject for s in splits[split]),
            label_images=sum(sequence_split[str(s)] == split and str(s).split('/')[2] == subject for s in seq),
            cached_images=sum(sequence_split[str(seq[i])] == split and str(seq[i]).split('/')[2] == subject
                              for i in np.flatnonzero(present)),
            complete_pairs=sum(sequence_split[s] == split and s.split('/')[2] == subject for s, f in complete))
            for subject in subjects}
    selected = {}
    summaries = {}
    old_holdout = json.loads((ROOT / 'runs/2026-11-17_sapiens-training-summary/evidence/data_same.json').read_text())['sequence']
    for split, budget in [('train', 2500), ('validation', 500)]:
        candidates = collections.defaultdict(list)
        for sequence, f in sorted(complete):
            if sequence_split[sequence] == split and not (split == 'train' and sequence == old_holdout):
                candidates[sequence].append(f)
        # Minimum source-frame-index separation is an engineering sampling rule.
        spaced = {}
        for s, frames in candidates.items():
            kept = []
            for f in frames:
                if not kept or f - kept[-1] >= 10:
                    kept.append(f)
            spaced[s] = kept
        by_subject = collections.defaultdict(dict)
        for s, frames in spaced.items():
            by_subject[s.split('/')[2]][s] = len(frames)
        quota = balanced_quota({s: sum(v.values()) for s, v in by_subject.items()}, budget)
        records = []
        for subject, n in quota.items():
            sq = balanced_quota(by_subject[subject], n)
            for s, k in sq.items():
                ids = np.linspace(0, len(spaced[s]) - 1, k, dtype=int) if k else []
                for j in ids:
                    f = spaced[s][int(j)]
                    for eye in ['left', 'right']:
                        i = complete[(s, f)][eye]
                        records.append(dict(source_index=i, sequence=s, subject=subject,
                                            frame=f, view=eye, pair_id=s + '/' + str(f),
                                            original_cache_key=str(paths[i])))
        selected[split] = records
        write(OUT / (split + '_selection.json'), dict(
            status='candidate sampling manifest; full decode and label-overlay checks pending',
            records=records))
        summaries[split] = dict(images=len(records), pairs=len(records)//2,
                                subject_pairs=quota, sequences=len({r['sequence'] for r in records}),
                                min_frame_index_gap=10,
                                selection_sha256=sha(OUT / (split + '_selection.json')))
    assert not {r['sequence'] for r in selected['train']} & {r['sequence'] for r in selected['validation']}
    assert old_holdout not in {r['sequence'] for r in selected['train']}
    # Decode the first selected pair for every subject, checking label bounds.
    mapping = json.loads((ROOT / 'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json').read_text())['joint_mapping']
    rw_ids = [m['rw_index'] for m in mapping if m['rw_index'] is not None]
    probes = []
    for split, records in selected.items():
        for subject in sorted({r['subject'] for r in records}):
            pair = next(r['pair_id'] for r in records if r['subject'] == subject)
            for r in [r for r in records if r['pair_id'] == pair]:
                key = r['original_cache_key']
                shard = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(shard_names)
                blob = connections[shard].execute('SELECT data FROM images WHERE key=?', (key,)).fetchone()[0]
                with Image.open(io.BytesIO(blob)) as im:
                    im.load()
                    w, h = im.size
                points = xy[r['source_index'], rw_ids, :2]
                finite = np.isfinite(points).all(axis=1)
                in_frame = finite & (points[:,0] >= 0) & (points[:,0] < w) & (points[:,1] >= 0) & (points[:,1] < h)
                probes.append(dict(split=split,subject=subject,sequence=r['sequence'],
                                   frame=r['frame'],view=r['view'],size=[w,h],
                                   decoded=True, finite_body_labels=int(finite.sum()),
                                   in_frame_body_labels=int(in_frame.sum()),image_sha256=hashlib.sha256(blob).hexdigest()))
    for c in connections.values():
        c.close()
    metadata = json.loads((DATA / 'prepared/metadata.json').read_text())
    old = json.loads((ROOT / 'runs/2026-11-16_sapiens-rw500-prep/evidence/archive_index.json').read_text())
    report = dict(command='python runs/2026-11-17_rw-a-scale-prep/audit.py', exit_status=0,
                  execution_baseline=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  official_split_and_cache_coverage=counts,
                  cache_manifest_sha256=sha(CACHE/'manifest.json'),cache_shards=shards,
                  samples_sha256=sha(DATA/'prepared/samples.npz'),
                  split_sha256=sha(DATA/'official_splits/UnrealEgoData_rw.zip'),
                  total_label_images=len(paths),cached_label_images=int(present.sum()),
                  complete_pairs=len(complete),selection=summaries,decode_probes=probes,
                  preserved_old_same_person_holdout=old_holdout,
                  sampling_revision='Initial candidate included the old local S6 holdout because it belongs to official train; final v2 excludes the entire sequence. Initial candidate retained locally and its audit retained as inventory_initial.json.',
                  original_roots_exist={k:Path(metadata[k]).exists() for k in ['source_root','rgb_root','pose_root']},
                  prior_partial_archive=dict(source='runs/2026-11-16_sapiens-rw500-prep/evidence/archive_index.json',
                    images_by_split=old['images_by_split'],archive_complete=old['archive_complete'],
                    error=old['archive_error'],fresh_archive_scan=False),
                  preliminary_probe_correction='Initial ad hoc cache lookup used SHA1 and missed; source reader inspection established MD5. All reported coverage uses MD5/index matches.',
                  limitations=['cache-key presence does not prove every image decodes',
                    'only one selected pair per train/validation subject decoded',
                    'original pose JSON unavailable; prepared labels not independently recertified',
                    'no test image cache coverage audited: source NPZ contains train/validation only',
                    'candidate selection only; no training launched'],
                  local_selection_directory='<WORKSPACE>/DownSight/outputs/rw_a_scale_prep_20261117_v2')
    write(RUN/'evidence/inventory.json', report)
    print(json.dumps(dict(selection=summaries,decoded_probes=len(probes),cache_images=int(present.sum())),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
