import json,re,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];run=Path(__file__).resolve().parent;base='5b5cf1d8d174de6a62e9e86f6ab4da3f0a5edccd'
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
mode='staged' if git('rev-parse','HEAD')==base else 'committed'
files=(git('diff','--cached','--name-only') if mode=='staged' else git('diff','--name-only',base,'HEAD')).splitlines();issues=[];rows=[]
patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}','-----'+r'BEGIN .*PRIVATE KEY'+'-----','/'+'vepfs-','/'+'home/',r'(^|[\s"=])'+'/'+'root/']
for f in files:
 if not f.startswith('runs/2026-11-15_auto-loss-comparison/'):issues.append('outside current run: '+f)
 data=subprocess.check_output(['git','show',(':' if mode=='staged' else 'HEAD:')+f],cwd=root);path=root/f
 if path.read_bytes()!=data:issues.append('working tree differs: '+f)
 if len(data)>=50_000_000 or path.suffix not in ['.py','.md','.txt','.json'] or path.name=='.env':issues.append('disallowed size or type: '+f)
 if any(re.search(pattern,data.decode('utf8')) for pattern in patterns):issues.append('sensitive pattern: '+f)
 rows.append({'path':f,'bytes':len(data)})
remote=git('ls-remote','origin','refs/heads/main').split()[0]
if remote!=base:issues.append('remote changed')
if subprocess.run(['git','diff','--cached','--check'],cwd=root).returncode:issues.append('staged whitespace check failed')
result={'base':base,'mode':mode,'remote_head':remote,'files':rows,'workspace_status':git('status','--short'),'errors':issues,'exit_status':int(bool(issues)),'scope':'Incremental selected files only; existing untracked work excluded. No model/data/video/raw output or full provenance.'}
if mode=='staged':(run/'evidence/prepush.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert files and not issues
