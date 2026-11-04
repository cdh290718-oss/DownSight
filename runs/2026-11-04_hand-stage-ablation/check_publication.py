"""Audit the exact pending repository increment and new static site."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worktree','staged','commit'],default='worktree');p.add_argument('--out',required=True,type=Path);a=p.parse_args()
base='d08ca4f736dfdf2284bf184d196e7e65a8a5339f'
cmd=['git','diff']+(['--cached'] if a.mode=='staged' else [])+['--name-only',base]+(['HEAD'] if a.mode=='commit' else [])
files=subprocess.check_output(cmd,text=True).splitlines()
if a.mode=='worktree':files+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
files=sorted(set(files));entries=[];fail=[]
excluded='runs/2026-11-04_session-hand6-v2/evidence/point_comparison_followup.json'
for name in files:
    path=ROOT/name
    if path.resolve()==a.out.resolve():continue
    content=path.read_bytes() if a.mode=='worktree' else subprocess.check_output(['git','show',(':' if a.mode=='staged' else 'HEAD:')+name])
    if len(content)>=50_000_000 or path.suffix in {'.npz','.npy','.pt','.pth','.mp4','.pkl','.ckpt','.safetensors'} or path.name.startswith('.env'):fail.append(name+': asset/size')
    if path.suffix not in {'.jpg','.png'}:
        text=content.decode()
        for pattern in ['/vepfs'+r'-',r'/root/(?:\.ssh|\.config|\.codex)',r'ghp_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY']:
            if re.search(pattern,text):fail.append(name+': path/credential pattern')
    if name==excluded and a.mode!='worktree':fail.append(name+': previous file must not be published')
    entries.append(dict(path=name,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),publish=name!=excluded))
site=[]
for path in sorted((RUN/'site').rglob('*')):
    if not path.is_file():continue
    content=path.read_bytes();relative=path.relative_to(RUN/'site').as_posix()
    if path.suffix not in {'.html','.json','.jpg'} or len(content)>=50_000_000:fail.append(relative+': site asset')
    if path.suffix=='.jpg':
        with Image.open(path) as im:
            assert im.size==(1280,720) and not im.getexif()
            im.verify()
    site.append(dict(path=relative,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content)))
html=(RUN/'site/index.html').read_text()
assert html.count('<img ')==18
for relative in re.findall(r'(?:href|src)="([^"]+)"',html):
    if relative.startswith(('https://','#')):continue
    assert (RUN/'site'/relative).is_file()
metrics=json.loads((RUN/'site/metrics.json').read_text())
assert len(metrics['frames'])==6 and all(set(r['methods'])=={'A','B','C'} for r in metrics['frames'])
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True).split()[0]
assert remote==base, 'Remote changed; review before publishing'
subprocess.run(['git','diff','--check',base],check=True)
result=dict(passed=not fail,mode=a.mode,base=base,remote_main=remote,scope='Pending increment plus site; pre-existing untracked file scanned but excluded; no historical audit',failures=fail,files=entries,site=site)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print('PASS' if not fail else 'FAIL',len(entries),'files',len(site),'site assets')
if fail:raise SystemExit(1)
