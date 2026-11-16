"""Wait for the batch, then evaluate and verify only after all fitting has exited."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_official_noise_v2_20261116';pid=int(sys.argv[1]);py=json.loads((ROOT/'outputs/synthetic_original_20261108/configs/left.json').read_text())['paths']['body_python']
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
while alive(pid):time.sleep(20)
path=OUT/'execution.jsonl';lines=path.read_text().splitlines() if path.exists() else []
res={'runner_pid_recorded':True,'expected_records':144,'actual_records':len(lines),'checks':[]}
if len(lines)==144:
 for name in ['evaluate.py','summarize.py','verify.py']:
  env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='');cmd=[py if name=='evaluate.py' else sys.executable,str(RUN/name)];p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True);redact=lambda s:s.replace(str(ROOT),'<WORKSPACE>').replace(str(ROOT.parent),'<WORKSPACE_PARENT>');res['checks'].append({'script':name,'exit_status':p.returncode,'stdout':redact(p.stdout)[-4000:],'stderr':redact(p.stderr)[-4000:]})
  if p.returncode:break
res['exit_status']=0 if len(lines)==144 and len(res['checks'])==3 and all(x['exit_status']==0 for x in res['checks']) else 1
(RUN/'evidence/finish_status.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
