"""Freeze identities, then stream the local test RGB archive once with gzip CRC verification."""
import collections,hashlib,io,subprocess,tarfile
import numpy as np
from PIL import Image
from paths import *

def select():
    z=np.load(SOURCE)
    seq,frames,sides,paths=[z[k] for k in ('source_sequence','frame_index','camera_side','image_path')]
    old=read(OLD/'test_manifest.json')['records']
    excluded=collections.defaultdict(set)
    for r in old:excluded[r['sequence']].add(r['frame'])
    # Also exclude the frames used to calibrate the archived stereo rigs.
    rigs=read(ROOT/'runs/2026-11-18_rw_frozen200_golden_d/evidence/preparation.json')
    by_seq=collections.defaultdict(dict)
    lookup={(str(s),int(f),str(v)):i for i,(s,f,v) in enumerate(zip(seq,frames,sides))}
    for s,f,v in lookup:
        if v=='left' and (s,f,'right') in lookup:
            if all(abs(f-k)>=10 for k in excluded[s]):
                by_seq[s][f]=(lookup[s,f,'left'],lookup[s,f,'right'])
    def h(s):return hashlib.sha256(('downsight-test500-v1/'+s).encode()).digest()
    rows=[]
    calibrations={(r['calibration_sequence'],r['calibration_frame']) for r in rigs['rigs']} if 'rigs' in rigs else set()
    for subject,quota in [('S1',167),('S9',167),('S11',166)]:
        sequences=sorted((s for s in by_seq if s.split('/')[2]==subject),key=h)
        pool={s:sorted((f for f in by_seq[s] if (s,f) not in calibrations),key=lambda f:h(s+'/'+str(f))) for s in sequences}
        chosen=collections.defaultdict(list);n=0
        while n<quota:
            before=n
            for s in sequences:
                if n==quota:break
                pick=next((f for f in pool[s] if all(abs(f-g)>=10 for g in chosen[s])),None)
                if pick is not None:chosen[s].append(pick);n+=1
            assert n>before
        for s in sorted(chosen):
            for f in sorted(chosen[s]):
                inds=by_seq[s][f]
                rows.append(dict(pair='p%03d'%(len(rows)+1),subject=subject,sequence=s,frame=f,indices=inds,
                    members=[str(paths[i]).split('UnrealEgoData_rw_test_rgb/',1)[1] for i in inds]))
    assert len(rows)==500 and len({(r['sequence'],r['frame']) for r in rows})==500
    write(OUT/'manifest.json',rows)
    write(RUN/'evidence/sample.json',dict(rows=[{k:v for k,v in r.items() if k!='members'} for r in rows],
           selection='fixed hash per subject, sequence round-robin, distance >=10 frame indices to previous test and each selected frame within sequence',
           source_sha256=sha(SOURCE),old_manifest_sha256=sha(OLD/'test_manifest.json'),manifest_sha256=sha(OUT/'manifest.json')))

def extract():
    rows=read(OUT/'manifest.json')
    wanted={member:r['pair']+'_'+side+'.png' for r in rows for member,side in zip(r['members'],('left','right'))}
    files=[BACK/'.unrealego_rw_repair_cache/UnrealEgoData_rw_test_rgb/studio.tar.gz000']+[OLD/name for name in ('part000-tail','part001','part002','part003')]
    cat=subprocess.Popen(['cat',*files],stdout=subprocess.PIPE)
    unzip=subprocess.Popen(['gzip','-dc'],stdin=cat.stdout,stdout=subprocess.PIPE)
    cat.stdout.close()
    found={}
    with tarfile.open(fileobj=unzip.stdout,mode='r|',bufsize=512*1024) as archive:
        for obj in archive:
            if obj.name not in wanted or not obj.isfile():continue
            name=wanted[obj.name];assert name not in found
            raw=archive.extractfile(obj).read()
            with Image.open(io.BytesIO(raw)) as im:
                im.load();assert im.size==(872,872) and im.mode=='RGB'
            (OUT/'images'/name).write_bytes(raw)
            found[name]=hashlib.sha256(raw).hexdigest()
            if len(found)%100==0:print('decoded',len(found),flush=True)
    while unzip.stdout.read(1<<20):pass
    assert unzip.wait()==0 and cat.wait()==0 and len(found)==1000==len(set(found.values()))
    assert set(found.values()).isdisjoint(r['image_sha256'] for r in read(OLD/'test_manifest.json')['records'])
    write(OUT/'images.json',found)

if __name__=='__main__':
    select();extract()
