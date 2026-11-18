"""Read local split RGB tar stream without changing archives or extracting all data."""
import os,json,tarfile,subprocess,time
from pathlib import Path
root=Path(os.environ['UNREAL_ROOT']);out=Path('outputs/cross_dataset100_20261118')
parts=sorted((root/'UnrealEgoData2_test_rgb').glob('CityPark_Showcase.tar.gz*'))
p=subprocess.Popen(['cat',*[str(x) for x in parts]],stdout=subprocess.PIPE)
rows=[];error=None;start=time.time()
try:
 with tarfile.open(fileobj=p.stdout,mode='r|gz') as t:
  for m in t:
   if m.isfile() and m.name.endswith('.png'):
    rows.append({'member':m.name,'size':m.size})
    if len(rows)%1000==0:print('RGB members',len(rows),'seconds',round(time.time()-start),flush=True)
except Exception as e:error=type(e).__name__+': '+str(e)
finally:
 p.stdout.close();status=p.wait()
r={'members':rows,'error':error,'cat_exit':status,'elapsed_seconds':time.time()-start,'parts':[{'name':x.name,'bytes':x.stat().st_size} for x in parts]}
(out/'unreal_rgb_index.json').write_text(json.dumps(r,indent=2));print('DONE',len(rows),error,status,flush=True)
