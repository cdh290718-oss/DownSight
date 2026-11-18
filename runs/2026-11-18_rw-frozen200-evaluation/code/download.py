"""Download missing byte ranges to a new output directory; preserve old cache."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--confirmed',action='store_true');a=p.parse_args()
if not a.confirmed:p.error('Obtain explicit user confirmation before downloading')
root=Path(__file__).resolve().parents[3];out=root/'outputs/rw_test_completion_20261117';out.mkdir(exist_ok=False);url='https://unrealego.mpi-inf.mpg.de/data/UnrealEgoData_rw_test_rgb/studio/studio.tar.gz';old=root.parent/'EgoSMPLX/.unrealego_rw_repair_cache/UnrealEgoData_rw_test_rgb/studio.tar.gz000';sizes=[10737418240]*3+[117428803];prefix=old.stat().st_size;assert prefix==7634717831
files=[]
for i,size in enumerate(sizes):
 start=prefix if i==0 else 0;target=out/('part000-tail' if i==0 else 'part%03d'%i);req=urllib.request.Request(url+('%03d'%i),headers={'Range':f'bytes={start}-{size-1}'})
 with urllib.request.urlopen(req,timeout=90) as response:
  assert response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{size-1}/{size}'
  h=hashlib.sha256();count=0
  with target.open('xb') as f:
   while True:
    b=response.read(1024*1024)
    if not b:break
    f.write(b);h.update(b);count+=len(b)
    if count%(1024**3)<len(b):print('part',i,'bytes',count,flush=True)
  assert count==size-start,(i,count,size-start)
 files.append(dict(part=i,size=count,sha256=h.hexdigest()))
(out/'download_complete.json').write_text(json.dumps(dict(files=files,source='official UnrealEgo-RW test RGB',original_cache_read_only=True),indent=2)+'\n');print('download completed',flush=True)
