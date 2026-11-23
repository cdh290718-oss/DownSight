"""Freeze sample identities before looking at any fitting result or floor quality."""
from pathlib import Path
import hashlib
import json

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/fixed_floor24_20261123'
PRIOR = ROOT/'runs/2026-11-23_fullbody_joint_candidates'
OLD = ROOT/'runs/2026-11-20_rw20000_floor12'
MANIFEST = {'test':ROOT/'outputs/rw_test_completion_20261117/test_manifest.json',
            'validation':ROOT/'outputs/rw_a20000_dataset_20261117/validation.json'}
def read(path):
    return json.loads(path.read_text())
def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def grouped(split):
    records={}
    for r in read(MANIFEST[split])['records']:
        pair=records.setdefault(r['pair_id'],{})
        assert r['view'] not in pair
        pair[r['view']]=r
    prefix='pair' if split=='test' else 'val'
    return {prefix+f'{i:03d}':records[k] for i,k in enumerate(sorted(records),1)}
def freeze():
    t,v=grouped('test'),grouped('validation')
    old=[r['pair'] for r in read(OLD/'evidence/renders_floor.json')['rows']]
    assert len(t)==100 and len(v)==200 and len(old)==12
    new=[]
    for sub in sorted({r['left']['subject'] for r in v.values()}):
        pool=[k for k in v if v[k]['left']['subject']==sub]
        def order(k):
            name='fixed-floor24-v1/'+v[k]['left']['pair_id']
            return hashlib.sha256(name.encode()).hexdigest()
        seen=set()
        for k in sorted(pool,key=order):
            seq=v[k]['left']['sequence']
            if seq not in seen:
                new.append(k)
                seen.add(seq)
            if len(seen)==3:
                break
    rows=[];seen=set();digests=set()
    for split,selection,data in [('test',old,t),('validation',sorted(new),v)]:
        for pair in selection:
            eyes=data[pair]
            l,r=[eyes[s] for s in ('left','right')]
            identity=l['pair_id']
            assert identity==r['pair_id'] and identity not in seen
            seen.add(identity)
            for eye in (l,r):
                assert sha(MANIFEST[split].parent/eye['image'])==eye['image_sha256'] and eye['image_sha256'] not in digests
                digests.add(eye['image_sha256'])
            rows.append(dict(split=split,pair=pair,subject=l['subject'],sequence=l['sequence'],frame=l['frame'],image_sha256=[l['image_sha256'],r['image_sha256']]))
    assert len(rows)==24 and len(digests)==48 and len(new)==12
    write(RUN/'evidence/sample.json',dict(rule='SHA-256 fixed-floor24-v1/source pair_id, first three distinct sequences per validation subject',rows=rows))
if __name__=='__main__':
    freeze()
