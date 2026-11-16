"""Verify exact conditions and sanitize execution evidence after fitting."""
import json,collections
from pathlib import Path
r=Path(__file__).resolve().parents[2];run=Path(__file__).resolve().parent;out=r/'outputs/unrealego_official_noise_v2_20261116';records=[json.loads(s) for s in (out/'execution.jsonl').read_text().splitlines()];expected={(p,f'gt_s{s}_seed{z}',m) for p in [f'body{i:02d}' for i in range(1,5)] for s in [1,3,5,8] for z in [20261115,20261116,20261117] for m in ['3d','mixed','2d']};actual={(x['pair'],x['condition'],x['method']) for x in records};assert actual==expected and len(records)==len(expected)
checks=[];fail=[]
for row in records:
 d=out/row['pair']/row['condition']; obs=json.loads((d/'observations.json').read_text()); fallback=sum(not p['accepted'] and any(p['valid']) for p in obs['points']);
 if row.get('reused_from'):assert row['method']=='mixed' and row['reused_from']=='3d' and fallback==0
 if row['status']!='ok':fail.append({k:row[k] for k in ['pair','condition','method','exit_status']})
for pair,condition in sorted({(x['pair'],x['condition']) for x in records}):
 rr=[x for x in records if x['pair']==pair and x['condition']==condition and x['status']=='ok'];opts=[json.loads((out/pair/condition/x.get('reused_from',x['method'])/'optimization.json').read_text()) for x in rr]
 if opts:
  same_basis=all(x['initialization']['basis']==opts[0]['initialization']['basis'] for x in opts);same_root=all(x['initial_global']==opts[0]['initial_global'] for x in opts);assert same_basis and same_root;checks.append({'pair':pair,'condition':condition,'same_basis':same_basis,'same_root':same_root,'rank':opts[0]['initialization']['retained_rank']})
clean=[]
for row in records:
 z=dict(row)
 if not row.get('reused_from'):
  log=(out/row['pair']/row['condition']/(row['method']+'.log')).read_text(errors='replace');lines=[v for v in log.splitlines() if v.startswith('D ') or v.startswith('DONE D ')];z['progress_excerpt']=lines[-3:]
  if row['status']!='ok':z['failure_tail_redacted']=[v.replace(str(r),'<WORKSPACE>').replace(str(r.parent),'<WORKSPACE_PARENT>') for v in log.splitlines()[-8:]]
 clean.append(z)
(run/'evidence/execution.json').write_text(json.dumps({'command':'python3 runs/2026-11-16_unrealego-official-noise/run.py','records':clean,'failures':fail},indent=2)+'\n');(run/'evidence/verification.json').write_text(json.dumps({'exit_status':0,'conditions':checks,'failures':fail,'record_count':len(records),'aliases':sum('reused_from' in x for x in records)},indent=2)+'\n');print('verified',len(records),'records; failures',len(fail))
import subprocess,sys
subprocess.run([sys.executable,str(run/'make_report.py')],cwd=r,check=True)
