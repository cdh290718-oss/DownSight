import json,re,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];here=Path(__file__).parent;base='5281634c1bac19a27d5bdf0130d395230521446e'
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
mode='staged' if git('rev-parse','HEAD')==base else 'committed'; names=(git('diff','--cached','--name-only') if mode=='staged' else git('diff','--name-only',base,'HEAD')).splitlines();bad=[];files=[]
patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}','-----'+r'BEGIN .*PRIVATE KEY'+'-----','/'+'vepfs-','/'+'home/',r'(^|[\s"=])'+'/'+'root/']
for n in names:
 path=root/n; data=subprocess.check_output(['git','show',(':' if mode=='staged' else 'HEAD:')+n],cwd=root)
 if not n.startswith(('runs/2026-11-16_unrealego-official-noise/','runs/2026-11-16_official-noise-review/')):bad.append(n+':outside scope')
 if path.read_bytes()!=data or len(data)>=50_000_000 or path.suffix not in ('.json','.py','.txt','.md') or path.name=='.env':bad.append(n+':size/type/diff')
 if any(re.search(p,data.decode('utf-8')) for p in patterns):bad.append(n+':sensitive')
 files.append({'path':n,'bytes':len(data)})
remote=git('ls-remote','origin','refs/heads/main').split()[0]
if remote!=base:bad.append('remote changed')
if subprocess.run(['git','diff','--cached','--check'],cwd=root).returncode:bad.append('diff check')
res={'base':base,'remote':remote,'mode':mode,'files':files,'workspace_status':git('status','--short'),'errors':bad,'exit_status':int(bool(bad)),'scope':'new experiment and review only; no weights, data, videos, arrays, complete provenance or patch'}
if mode=='staged':(here/'evidence/prepush.json').write_text(json.dumps(res,indent=2)+'\n')
print('audit',mode,'files',len(files),'errors',bad);assert names and not bad
