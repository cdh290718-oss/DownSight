"""Verify that GitHub Pages serves every audited asset byte for byte."""
import argparse,hashlib,json,time,urllib.request,urllib.error
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--timeout',type=int,default=600);a=p.parse_args()
receipt=a.evidence/'publication.json';publication=json.loads(receipt.read_text());deadline=time.monotonic()+a.timeout
checks=[]
for name,digest in publication['files'].items():
 url=publication['url']+name+'?commit='+publication['commit'][:12]
 while True:
  try:
   with urllib.request.urlopen(url,timeout=30) as response:
    raw=response.read();status=response.status;mime=response.headers.get('Content-Type')
   actual=hashlib.sha256(raw).hexdigest()
   if status==200 and actual==digest:break
   reason='HTTP '+str(status)+' or content digest differs'
  except (urllib.error.URLError,TimeoutError) as e:reason=str(e)
  if time.monotonic()>=deadline:raise RuntimeError(name+': '+reason)
  print('Waiting for Pages:',name,reason,flush=True);time.sleep(10)
 checks.append(dict(path=name,http_status=status,sha256=actual,expected_sha256=digest,bytes=len(raw),content_type=mime))
 print('VERIFIED',name,flush=True)
(a.evidence/'online_verification.json').write_text(json.dumps(dict(passed=True,url=publication['url'],commit=publication['commit'],files=checks),indent=2)+'\n')
publication['online_verified']=True;receipt.write_text(json.dumps(publication,indent=2)+'\n')
