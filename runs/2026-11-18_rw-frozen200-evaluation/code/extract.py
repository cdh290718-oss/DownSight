"""Stream sequential gzip parts; extract only preselected images to new output."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parents[3];out=root/'outputs/rw_test_completion_20261117';old=root.parent/'EgoSMPLX/.unrealego_rw_repair_cache/UnrealEgoData_rw_test_rgb/studio.tar.gz000';selection=root/'outputs/rw_a_budget_20261117/test_selection.json';expected='6bba850b8cf580704d4111b797792dfc75ea23aa8166a51ad4c008c8fa53b6e6';assert hashlib.sha256(selection.read_bytes()).hexdigest()==expected
files=[old,out/'part000-tail',out/'part001',out/'part002',out/'part003'];assert (out/'download_complete.json').exists()
check=subprocess.Popen(['cat',*[str(p) for p in files]],stdout=subprocess.PIPE);validation=subprocess.run(['gzip','-t'],stdin=check.stdout);check.stdout.close();assert check.wait()==0 and validation.returncode==0, 'gzip integrity validation failed'
rows=json.loads(selection.read_text())['records'];wanted={r['original_cache_key'].split('UnrealEgoData_rw_test_rgb/',1)[1]:r for r in rows};assert len(wanted)==200;dest=out/'images';dest.mkdir(exist_ok=False);found={}
proc=subprocess.Popen(['cat',*[str(p) for p in files]],stdout=subprocess.PIPE,stderr=subprocess.PIPE);error=None
try:
 with tarfile.open(fileobj=proc.stdout,mode='r|gz') as tar:
  for member in tar:
   if member.name not in wanted or not member.isfile():continue
   r=wanted[member.name];raw=tar.extractfile(member).read()
   with Image.open(io.BytesIO(raw)) as image:image.load();width,height=image.size
   name=hashlib.sha256(member.name.encode()).hexdigest()+'.png';(dest/name).open('xb').write(raw);found[member.name]=dict(image='images/'+name,image_sha256=hashlib.sha256(raw).hexdigest(),sequence=r['sequence'],subject=r['subject'],frame=r['frame'],view=r['view'],pair_id=r['pair_id'],source_index=r['source_index'],size=[width,height])
except (tarfile.TarError,EOFError,OSError) as e:error=repr(e)
finally:
 proc.stdout.close();proc.wait()
 if proc.returncode!=0 and error is None:error='stream command failed: '+str(proc.returncode)
pairs={}
for r in found.values():pairs.setdefault(r['pair_id'],set()).add(r['view'])
result=dict(requested=len(wanted),found=len(found),complete_pairs=sum(v=={'left','right'} for v in pairs.values()),missing=sorted(set(wanted)-set(found)),error=error,records=list(found.values()))
(out/'extraction.json').write_text(json.dumps(result,indent=2)+'\n');print('found',len(found),'pairs',result['complete_pairs'],'error',error,flush=True)
