"""Keep scalar execution evidence and failures; full provenance stays local."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();repo=Path.cwd();workspace=repo.parent

def redact(text):return text.replace(str(repo),'<REPO>').replace(str(workspace),'<WORKSPACE>')
def save(name,text):(a.evidence/name).write_text(redact(text).rstrip()+'\n')
inputs=json.loads((a.root/'prepared/inputs.json').read_text());execution=[]
configurations=[]
for r in inputs['records']:
 cfg=json.loads(Path(r['config']).read_text())
 configurations.append(dict(frame=r['id'],config_sha256=hashlib.sha256(Path(r['config']).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(Path(cfg['manifest']).read_bytes()).hexdigest(),context_sha256=hashlib.sha256(Path(r['context']).read_bytes()).hexdigest(),settings={k:v for k,v in cfg.items() if k not in ['paths','manifest','output']},resource_identifiers={k:Path(v).name for k,v in cfg['paths'].items()}))
save('config_summary.json',json.dumps(configurations,indent=2)+'\n')
for d in [a.root/'batch',a.root/'batch_cpu']:
 if (d/'execution.json').exists():execution.extend(json.loads((d/'execution.json').read_text()))
save('execution.json',json.dumps(execution,indent=2)+'\n')
for name in ['pytest-initial.txt','pytest-tests.txt','pytest-final.txt','pytest-published.txt']:
 if (a.root/name).exists():save(name,(a.root/name).read_text())
code=json.loads((a.root/'batch_cpu/code_hashes.json').read_text())
save('source_hashes.json',json.dumps(dict(base_commit='21cdefc2a9cb58cf3dbc4072ec878742a4b90ad2',worktree_modified_at_execution=True,pipeline=code),indent=2)+'\n')
# Keep all failure tracebacks; keep complete progress/constraint lines in successful worker logs.
for logfile in a.root.rglob('*.log'):
 text=logfile.read_text(errors='replace');lines=text.splitlines();failure=any(s in text for s in ('Traceback (most recent call last)','AssertionError','RuntimeError','NameError','ValueError'))
 if failure:excerpt=text
 else:excerpt='\n'.join(line for line in lines if any(s in line for s in ['RUN ','FRAME_COMPLETE','VERIFIED','DONE','SEAM','FINAL','FUSED','rejects',' step','Source reconstruction','\'loss\':','\'step\':']))
 if excerpt:
  name='log_'+str(logfile.relative_to(a.root)).replace('/','__').replace('.log','.txt');save(name,excerpt+'\n')
# Descriptive asset IDs, digests and input summaries, without local paths.
for r in inputs['records']:
 cfg=json.loads(Path(r['config']).read_text());out=Path(cfg['output']);pre=out/'preflight.json'
 if pre.exists():
  q=json.loads(pre.read_text());tree=q['backend']['source_tree'];inventory='backend_source_'+tree['tree_sha256'][:12]+'.json'
  save(inventory,json.dumps(tree,indent=2)+'\n')
  tree.pop('files');tree['published_inventory']=inventory;q['local_preflight_sha256']=hashlib.sha256(pre.read_bytes()).hexdigest()
  save(r['id']+'_preflight.json',json.dumps(q,indent=2)+'\n')
  prov=sorted((out/'provenance').glob('*.json'))
  summary=[]
  for pp in prov:
   q=json.loads(pp.read_text());g=q['pipeline_git'];summary.append(dict(file=pp.name,sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),keys=sorted(q),pipeline_commit=g['commit'],pipeline_dirty=g['dirty'],pipeline_files_sha256=q['pipeline_files_sha256'],started_utc=q['started_utc'],finished_utc=q.get('finished_utc'),state=q['state'],python=q['python'],platform=q['platform']))
  save(r['id']+'_provenance_summary.json',json.dumps(summary,indent=2)+'\n')
print('Collected',len(list(a.evidence.glob('*'))),'evidence files')
