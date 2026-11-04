"""Check exact public increment; does not claim to audit historical contents."""
import argparse,json,re,subprocess,hashlib,math
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['worktree','staged','commit'],default='worktree');p.add_argument('--base',default='HEAD');p.add_argument('--ref',default='HEAD');a=p.parse_args();repo=Path.cwd()
diff=['git','diff']+(['--cached'] if a.mode=='staged' else [])+['--name-only',a.base]+([a.ref] if a.mode=='commit' else [])
tracked=subprocess.check_output(diff,text=True).splitlines();new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines() if a.mode=='worktree' else [];files=sorted(set(tracked+new));fail=[];result=[]
blocked={'.npz','.npy','.mp4','.avi','.pt','.pth','.ckpt','.safetensors','.pkl','.zip'}
for relative in files:
 f=repo/relative
 if a.mode=='worktree' and f.resolve()==a.out.resolve():continue
 if a.mode=='worktree':
  if not f.is_file():fail.append(relative+': not a regular file');continue
  data=f.read_bytes()
 else:data=subprocess.check_output(['git','show',(':' if a.mode=='staged' else a.ref+':')+relative])
 if len(data)>=50_000_000 or f.suffix in blocked or f.name.startswith('.env'):fail.append(relative+': prohibited asset or size')
 if f.suffix not in {'.jpg','.png','.svg'}:
  text=data.decode('utf-8')
  for pattern in ['/vepfs'+r"-[^\s\"'<>]+",r'/root/(?:\.config|\.ssh|\.codex)',r'ghp_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY']:
   if re.search(pattern,text):fail.append(relative+': sensitive/path pattern')
 if 'provenance' in f.name and 'summary' not in f.name and 'redacted' not in f.name and f.suffix=='.json':fail.append(relative+': full provenance not allowed')
 result.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
site=[]
for f in a.site.rglob('*'):
 if f.is_file():
  if f.stat().st_size>=50_000_000 or f.suffix not in {'.html','.json','.jpg','.css','.js'}:fail.append(str(f)+': bad public site asset')
  site.append(dict(path=f.relative_to(a.site).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  if f.suffix=='.jpg':
   with Image.open(f) as im:
    if im.size!=(1280,720) or im.getexif():fail.append(str(f)+': unexpected dimensions or EXIF')
    im.verify()
metrics=json.loads((a.site/'metrics.json').read_text())
if len(metrics['frames'])!=10 or any(set(r['methods'])!={'original','v2'} or any(v.get('failed') for v in r['methods'].values()) for r in metrics['frames']):fail.append('expected all ten completed view comparisons')
def finite(value):
 if isinstance(value,float):return math.isfinite(value)
 if isinstance(value,dict):return all(finite(v) for v in value.values())
 if isinstance(value,list):return all(finite(v) for v in value)
 return True
if not finite(metrics):fail.append('nonfinite public metric')
html=(a.site/'index.html').read_text()
for relative in re.findall(r'(?:src|href)="([^"]+)"',html):
 if relative.startswith(('https://','#','../')):continue
 if not (a.site/relative).is_file():fail.append('missing link '+relative)
if len(re.findall('<img ',html))!=20:fail.append('expected 20 comparison images')
subprocess.run(['git','diff']+(['--cached'] if a.mode=='staged' else [])+['--check',a.base]+([a.ref] if a.mode=='commit' else []),check=True)
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True).strip();status=subprocess.check_output(['git','status','--short'],text=True)
q=dict(passed=not fail,scope=a.mode+' increment and site; no historical audit; working-tree output audit excludes its own changing file',base=a.base,ref=a.ref if a.mode=='commit' else None,failures=fail,files=result,site=site,remote=remote,status=status)
a.out.write_text(json.dumps(q,indent=2)+'\n');print('PASS' if q['passed'] else 'FAIL',len(result),'repository files',len(site),'site files');print('\n'.join(fail))
if fail:raise SystemExit(1)
