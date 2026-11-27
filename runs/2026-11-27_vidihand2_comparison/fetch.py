"""Fetch pinned inference sources and range-verified model artifacts into local outputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import requests

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/vidihand2_comparison_20261127'
VENDOR=OUT/'vendor'
PIN='23cf80c292f2fb2db20caa1dee96516a63ee30a7'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    weights=json.loads((RUN.parent/'2026-11-27_vidihand2_wrist/evidence/weights.json').read_text())
    paths=['utils/io.py','utils/mano.py','vidihand_v2/config.yaml','vidihand_v2/inference.py','vidihand_v2/archs/arch_utils.py','vidihand_v2/archs/vidihand2_arch.py','vidihand_v2/archs/wan.py']
    base='https://raw.githubusercontent.com/NTUYWANG103/ViDiHand/'+PIN+'/'
    def source(name):
        p=VENDOR/name
        p.parent.mkdir(parents=True,exist_ok=True)
        r=requests.get(base+name,timeout=25)
        r.raise_for_status()
        p.write_bytes(r.content)
        return name,digest(p)
    with ThreadPoolExecutor(6) as ex: sources=dict(ex.map(source,paths))
    checks={}
    for f in weights['files']:
        p=VENDOR/'checkpoints'/f['path']
        p.parent.mkdir(exist_ok=True,parents=True)
        if not p.exists():
            url='https://huggingface.co/WANG-XIXI/ViDiHand/resolve/'+weights['revision']+'/'+f['path']
            partial=p.with_suffix('.partial')
            length=f['size']
            block=16<<20
            markers=p.parent/('.completed_'+f['sha256'])
            markers.mkdir(exist_ok=True)
            resume=partial.exists() and partial.stat().st_size==length
            with partial.open('r+b' if resume else 'w+b') as dst:
                dst.truncate(length)
                offsets=[start for start in range(0,length,block)
                         if not resume or not (markers/str(start)).is_file()]
                print('remaining ranges',len(offsets),flush=True)
                def grab(start):
                    end=min(length,start+block)-1
                    for attempt in range(6):
                        try:
                            with requests.get(url+f'?part={start}&retry={attempt}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(15,60)) as resp:
                                assert resp.status_code==206 and resp.headers['Content-Range']==f'bytes {start}-{end}/{length}',(resp.status_code,resp.headers.get('Content-Range'))
                                offset=start
                                for chunk in resp.iter_content(1<<20):
                                    assert os.pwrite(dst.fileno(),chunk,offset)==len(chunk)
                                    offset+=len(chunk)
                                assert offset==end+1
                                (markers/str(start)).touch()
                            return
                        except Exception:
                            if attempt==5: raise
                            time.sleep(attempt+1)
                with ThreadPoolExecutor(48) as ex:
                    for idx,_ in enumerate(ex.map(grab,offsets)):
                        if idx%32==0: print(f['path'],idx*block,flush=True)
            if digest(partial)!=f['sha256']:
                raise ValueError('Incomplete or corrupt artifact; discard the partial and retry')
            partial.rename(p)
        assert p.stat().st_size==f['size'] and digest(p)==f['sha256']
        checks[f['path']]=dict(bytes=f['size'],sha256=f['sha256'])
    (RUN/'evidence/download.json').write_text(json.dumps(dict(source_commit=PIN,source_files=sources,weights_revision=weights['revision'],weights=checks),indent=2)+'\n')
if __name__=='__main__': main()
