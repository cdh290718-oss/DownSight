"""Revise evaluation candidates and estimate runtime from existing logs only."""
import collections
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/rw_a_budget_20261117'
PREVIOUS = ROOT / 'outputs/rw_a_scale_prep_20261117_v2'
BACKEND = ROOT.parent / 'EgoSMPLX'


def read(p):
    return json.loads(p.read_text())


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def select(records, subject_quota):
    rng = np.random.default_rng(20261117)
    grouped = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in records:
        grouped[r['subject']][r['sequence']].setdefault(r['frame'], []).append(r)
    result = []
    for subject, quota in subject_quota.items():
        sequences = sorted(grouped[subject])
        rng.shuffle(sequences)
        amounts = dict.fromkeys(sequences, 0)
        capacity = {s: len(grouped[subject][s]) for s in sequences}
        assert sum(capacity.values()) >= quota
        while sum(amounts.values()) < quota:
            for s in sequences:
                if amounts[s] < capacity[s] and sum(amounts.values()) < quota:
                    amounts[s] += 1
        for s, n in amounts.items():
            frames = sorted(grouped[subject][s])
            for j in np.linspace(0, len(frames)-1, n, dtype=int) if n else []:
                result.extend(grouped[subject][s][frames[int(j)]])
    return result


def main():
    OUT.mkdir(exist_ok=False)
    validation = select(read(PREVIOUS/'validation_selection.json')['records'],
                        {s: 50 for s in ['S3','S5','S13','S15']})
    excluded = {r['sequence'] for r in read(ROOT/'runs/2026-11-17_2d-intervention/evidence/preparation.json')['pairs']}
    with np.load(BACKEND/'data/UnrealEgo_RW_Test/prepared/samples.npz', allow_pickle=False) as z:
        seq, frames, views, paths = [z[k] for k in ['source_sequence','frame_index','camera_side','image_path']]
    pairs = collections.defaultdict(dict)
    for i, s in enumerate(seq):
        s = str(s)
        if s not in excluded:
            pairs[(s, int(frames[i]))][str(views[i])] = i
    candidates = []
    last = {}
    for (s, f), vv in sorted(pairs.items()):
        if set(vv) != {'left','right'} or (s in last and f-last[s] < 10):
            continue
        last[s] = f
        for eye in ['left','right']:
            i = vv[eye]
            candidates.append(dict(source_index=i,sequence=s,subject=s.split('/')[2],frame=f,
                                   view=eye,pair_id=s+'/'+str(f),original_cache_key=str(paths[i])))
    test = select(candidates, {'S1':34,'S9':33,'S11':33})
    cache = BACKEND/'data/image_cache/egosmplx_training_safe_20260726'
    cm = read(cache/'manifest.json'); connections = {}
    summaries = {}
    for split, records in [('validation', validation), ('test', test)]:
        hits = collections.Counter()
        for r in records:
            key = r['original_cache_key']
            shard = int(hashlib.md5(key.encode()).hexdigest(),16) % len(cm['shards'])
            if shard not in connections:
                connections[shard] = sqlite3.connect((cache/cm['shards'][shard]).as_uri()+'?mode=ro',uri=True)
            hit = connections[shard].execute('SELECT 1 FROM images WHERE key=?',(key,)).fetchone() is not None
            r['available_in_training_sqlite_cache'] = hit
            hits[r['subject']] += int(hit)
        p = OUT/(split+'_selection.json')
        write(p,dict(status='candidate; not a completed independent evaluation',records=records))
        summaries[split] = dict(images=len(records),pairs=len(records)//2,
            subject_images=dict(collections.Counter(r['subject'] for r in records)),
            sequences=len({r['sequence'] for r in records}),cache_hits=dict(hits),
            sha256=sha(p),source='previous validation selection' if split=='validation' else 'test label metadata only')
        grouped = collections.defaultdict(list)
        for r in records: grouped[r['pair_id']].append(r['view'])
        assert all(sorted(v)==['left','right'] for v in grouped.values())
    for c in connections.values(): c.close()
    assert len(validation)==400 and len(test)==200
    assert not {r['sequence'] for r in test} & excluded
    train = read(PREVIOUS/'train_selection.json')['records']
    assert not {r['sequence'] for r in train} & {r['sequence'] for r in validation+test}
    prior = read(ROOT/'runs/2026-11-17_rw-a-scale-prep/evidence/inventory.json')
    capacity = {s: dict(images=sum(v['cached_images'] for v in group.values()),
                       pairs=sum(v['complete_pairs'] for v in group.values()),subjects=len(group))
                for s, group in prior['official_split_and_cache_coverage'].items() if s!='test'}
    timing_source = ROOT/'runs/2026-11-17_sapiens-training-summary/evidence/full_training.json'
    t = read(timing_source)
    # t0 in full_train.py is after loading the model and creating the optimizer.
    # Subsequent evaluation elapsed_seconds are cumulative, not per-evaluation durations.
    total = t['status']['elapsed_seconds']; eval100 = t['evaluations'][0]['elapsed_seconds']
    proxy = total - len(t['evaluations'])*eval100
    proposed = proxy + 2*4*eval100
    estimate = dict(source=str(timing_source.relative_to(ROOT)),source_sha256=sha(timing_source),
        command='Arithmetic extrapolation from full_training.json; no new timed training',
        previous_images_per_epoch=t['config']['train_images'],previous_epochs=t['config']['epochs'],
        previous_optimizer_steps=t['status']['step'],previous_measured_loop_seconds=total,
        previous_initial_100_image_validation_seconds=eval100,
        previous_validation_passes=len(t['evaluations']),
        proposed_training_images=5000,proposed_epochs=1,proposed_optimizer_steps=1250,
        proposed_validation_images=400,proposed_validation_passes=2,
        rough_training_and_saving_proxy_seconds=proxy,rough_new_loop_seconds=proposed,
        planning_range_minutes=[60,90],
        assumptions=['same single A100 80GB, precision, resolution, effective batch and full-backbone training',
                     'use initial validation duration as a proxy for all previous validation passes',
                     'proxy retains previous checkpoint-saving overhead; not pure training throughput',
                     'no large competing GPU workload; expanded-cache I/O may differ'],
        excludes=['dataset audit and training-loader implementation','model load before timer starts',
                  'additional checkpoint comparisons','final test and full-2D D fitting'],
        no_new_training=True)
    write(RUN/'evidence/timing.json',estimate)
    write(RUN/'evidence/selection.json',dict(command='python '+str(Path(__file__).relative_to(ROOT)),exit_status=0,
        capacity=capacity,selection=summaries,training_selection_unchanged_sha256=sha(PREVIOUS/'train_selection.json'),
        training_sample_percent=5000/capacity['train']['images']*100,
        excluded_previous_S9_sequences=sorted(excluded),
        test_status='Label-based candidates only; missing in inspected training SQLite cache. Partial test archive exists, but coverage of all three subjects remains unverified. No fresh/blind-test claim.',
        cache_scope='train/validation cache key presence, not all-image decode; test result only for selected candidates',
        preliminary_lookup_failure='Initial rg used two nonexistent training filenames and exited 2; corrected to code/full_train.py; no execution failed.',
        local_output='<WORKSPACE>/DownSight/outputs/rw_a_budget_20261117'))
    gpu = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    write(RUN/'evidence/gpu.json',dict(command='nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader',exit_status=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr,note='snapshot only; no reservation or guarantee of future availability'))
    print(json.dumps(dict(capacity=capacity,selection=summaries,estimated_minutes=proposed/60),ensure_ascii=False))


if __name__ == '__main__':
    main()
