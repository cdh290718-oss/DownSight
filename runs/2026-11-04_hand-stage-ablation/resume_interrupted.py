"""Retain interrupted logs and resume only unfinished views after turn abort."""
import json,os,shutil,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/hand_stage_ablation_20261104/C'
read=lambda p:json.loads(p.read_text())
records=read(OUT/'inputs.json')['records'];pending=[];receipt=[]
backup=OUT/'interrupted_attempt';backup.mkdir(exist_ok=False)
for r in records:
    c=read(Path(r['config']));root=Path(c['output'])
    if read(root/'run_state.json')['state']=='complete':continue
    dest=backup/r['id'];dest.mkdir()
    logs={}
    for p in root.rglob('*.log'):
        relative=p.relative_to(root);target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        logs[str(relative)]=p.read_text(errors='replace').splitlines()[-5:]
    shutil.copy2(root/'run_state.json',dest/'run_state.json')
    receipt.append(dict(frame=r['id'],reason='Conversation turn aborted; executor session disappeared and no worker process remained',interrupted_exit_status=None,last_logs=logs))
    pending.append((r,c))
assert len(pending)==2
def redact(x):
    if isinstance(x,str):return x.replace(str(ROOT.parent),'<WORKSPACE>').replace('/opt'+'/conda','<BUILD_ROOT>')
    if isinstance(x,list):return [redact(v) for v in x]
    if isinstance(x,dict):return {k:redact(v) for k,v in x.items()}
    return x
(RUN/'evidence/interruption.json').write_text(json.dumps(redact(receipt),indent=2)+'\n')
def execute(item):
    r,c=item;cmd=[c['paths']['body_python'],'-m','egosmplx_pipeline','run','--config',r['config'],'--cached-predictions','--resume'];start=time.time()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),TMPDIR=str(OUT/'batch'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (OUT/'batch'/(r['id']+'.resume.log')).open('w') as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    return dict(id=r['id'],pair=r['pair'],command=cmd,exit_status=result.returncode,elapsed_seconds=time.time()-start,output=c['output'],resumed_after_interruption=True)
finished=read(OUT/'batch/execution.json')
with ThreadPoolExecutor(max_workers=2) as pool:
    for f in as_completed([pool.submit(execute,r) for r in pending]):
        q=f.result();finished.append(q);(OUT/'batch/execution.json').write_text(json.dumps(finished,indent=2)+'\n');print(q['id'],q['exit_status'],round(q['elapsed_seconds'],1),flush=True)
