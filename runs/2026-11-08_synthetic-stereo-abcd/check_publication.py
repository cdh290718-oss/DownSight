"""Audit working tree, staged increment or commit, without auditing history."""
import argparse,hashlib,json,re,subprocess
from io import BytesIO
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
BASE='118c485f5b73c5767d050983d371a0315a9e7dba'
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worktree','staged','commit'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
args=['git','diff']+(['--cached'] if a.mode=='staged' else [])+['--name-only',BASE]+(['HEAD'] if a.mode=='commit' else [])
files=subprocess.check_output(args,text=True,cwd=ROOT).splitlines()
if a.mode=='worktree':files+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True,cwd=ROOT).splitlines()
files=sorted(set(files));entries=[];errors=[]
exclude='runs/2026-11-04_session-hand6-v2/evidence/point_comparison_followup.json'
for name in files:
    p=ROOT/name
    if p.resolve()==a.out.resolve():continue
    if a.mode=='worktree':
        if not p.is_file():errors.append(name+': not a file');continue
        raw=p.read_bytes()
    else:raw=subprocess.check_output(['git','show',(':' if a.mode=='staged' else 'HEAD:')+name],cwd=ROOT)
    publish=name!=exclude
    allowed_code=set()
    if a.mode!='worktree' and not name.startswith('runs/2026-11-08_synthetic-stereo-abcd/') and name not in allowed_code:errors.append(name+': outside this run')
    if len(raw)>=50000000 or p.suffix.lower() in {'.npz','.npy','.mp4','.avi','.pth','.pt','.ckpt','.pkl','.safetensors'} or p.name.startswith('.env'):errors.append(name+': prohibited asset or size')
    if p.suffix.lower() not in {'.jpg','.png'}:
        s=raw.decode('utf-8')
        for pattern in ['/vepfs'+r'-',r'/root/(?:\.ssh|\.config|\.codex)','/opt'+'/conda/',r'ghp_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY',r'(?i)(?:password|api_key|access_token|authorization)\s*[:=]\s*["\']?[A-Za-z0-9_./+~-]{8,}']:
            if re.search(pattern,s):errors.append(name+': machine path or credential')
    else:
        with Image.open(BytesIO(raw)) as im:
            if im.size!=(1280,720) or im.getexif():errors.append(name+': dimensions or EXIF')
            im.verify()
    if name==exclude and a.mode!='worktree':errors.append(name+': unrelated pre-existing file staged')
    if 'provenance' in p.name and p.suffix.lower()=='.json':errors.append(name+': complete provenance not allowed')
    entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),publish=publish))
if (RUN/'projection.md').exists():
    for relative in re.findall(r'!\[[^]]*\]\(([^)]+)\)',(RUN/'projection.md').read_text()):
        if not (RUN/relative).is_file():errors.append('missing projection asset '+relative)
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,cwd=ROOT).split()[0]
if remote!=BASE:errors.append('Remote main changed; inspect and reconcile before push')
subprocess.run(['git','diff','--check',BASE],cwd=ROOT,check=True)
result=dict(passed=not errors,mode=a.mode,base=BASE,remote_main=remote,scope='Current increment; pre-existing untracked file scanned but excluded; no historical audit',errors=errors,files=entries)
a.out.write_text(json.dumps(result,indent=2)+'\n');print('PASS' if not errors else 'FAIL',len(entries),'files',errors)
if errors:raise SystemExit(1)
