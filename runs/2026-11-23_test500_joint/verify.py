"""Audit 500 frozen test identities, re-pool frame scores, and check prior-frame exclusions."""
import collections,statistics
from paths import *
def main():
    r=read(RUN/'evidence/results.json')
    s=read(RUN/'evidence/sample.json')
    samples=s['rows']; results=r['rows']
    prior={(v['sequence'],v['frame']) for v in read(OLD/'test_manifest.json')['records']}
    prior_val={(v['sequence'],v['frame']) for v in read(ROOT/'outputs/rw_a20000_dataset_20261117/validation.json')['records']}
    ids={(x['sequence'],x['frame']) for x in samples}
    assert len(ids)==len(samples)==500 and not ids&(prior|prior_val)
    same=collections.defaultdict(list)
    for seq,frame in ids:same[seq].append(frame)
    spacing=min(f-g for fs in same.values() for g,f in zip(sorted(fs),sorted(fs)[1:]))
    gap=min(abs(f-g) for seq,f in ids for name,g in prior if seq==name)
    assert min(gap,spacing)>=10
    assert set(x['pair'] for x in samples)==set(results)|{x['pair'] for x in r['failures']}
    assert r['n_scored']==len(results) and set(results).isdisjoint(x['pair'] for x in r['failures'])
    orig={v['subject'] for v in samples}
    train=set(read(ROOT/'runs/2026-11-17_rw-a20000-training/evidence/dataset.json')['summary']['train']['subjects'])
    assert orig.isdisjoint(train)
    for method,vals in r['summary'].items():
        for k,v in vals.items():
            assert abs(statistics.mean(x['scores'][method][k] for x in results.values())-v)<1e-8
            for subject,sub in r['subjects'].items():
                group=[x['scores'][method][k] for x in results.values() if x['subject']==subject]
                assert len(group)==sub['n'] and abs(statistics.mean(group)-sub['summary'][method][k])<1e-8
    out=RUN/'evidence/verification.json'
    write(out,dict(status=0,source_pairs=len(ids),subjects=sorted(orig),sequences=len(same),
              same_sequence_min_frame_gap=spacing,prior_test_same_sequence_min_frame_gap=gap,
              previous_test_pair_overlap=len(prior&ids),previous_validation_pair_overlap=len(prior_val&ids),
              detector_training_subject_overlap=sorted(orig&train),per_frame_means_verified=True,
              results_sha256=sha(RUN/'evidence/results.json'),samples_sha256=sha(RUN/'evidence/sample.json')))
if __name__=='__main__':main()
