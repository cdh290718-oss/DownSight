"""Run each fixed view independently, retaining every failure and exit status."""
import argparse,json,os,subprocess,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--skip',default='');p.add_argument('--workers',type=int,default=1);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);records=json.loads(a.inputs.read_text())['records'];run=[]
for r in records:
 if r['id'] in a.skip.split(','):continue
 c=json.loads(Path(r['config']).read_text());run.append((r,c))
code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('egosmplx_pipeline').rglob('*.py')};(a.out/'code_hashes.json').write_text(json.dumps(code,indent=2)+'\n')
def execute(item):
 r,c=item;cmd=[c['paths']['body_python'],'-m','egosmplx_pipeline','run','--config',r['config'],'--cached-predictions'];start=time.time();env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(a.out.resolve()),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (a.out/(r['id']+'.log')).open('w') as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 return dict(id=r['id'],pair=r['pair'],command=cmd,exit_status=result.returncode,elapsed_seconds=time.time()-start,output=c['output'])
finished=[]
with ThreadPoolExecutor(max_workers=a.workers) as pool:
 for f in as_completed([pool.submit(execute,r) for r in run]):
  result=f.result();finished.append(result);(a.out/'execution.json').write_text(json.dumps(finished,indent=2)+'\n');print(result['id'],result['exit_status'],round(result['elapsed_seconds'],1),flush=True)
