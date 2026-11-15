import os,json,subprocess,concurrent.futures,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_gt16_20261115';python=json.loads((ROOT/'outputs/synthetic_original_20261108/configs/left.json').read_text())['paths']['body_python']
def alive(pid):
 try:
  p=Path('/proc')/str(pid)/'stat';return p.exists() and p.read_text().split()[2]!='Z'
 except (OSError,IndexError):return False
def one(pair,pid=None):
 dest=OUT/pair;start=time.time()
 if pid:
  while alive(pid):time.sleep(3)
  status=0 if (dest/'D_left/optimization.json').exists() else 1
 else:
  env=os.environ.copy();env.update(D_BATCH_PAIR=pair,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(dest/'runtime'),XDG_CACHE_HOME=str(dest/'runtime/cache'),MPLCONFIGDIR=str(dest/'runtime/matplotlib'));cmd=[python,str(RUN/'fit.py'),'--case','D']
  with (dest/'D.log').open('x') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  status=r.returncode
 return {'pair':pair,'method':'D','exit_status':status,'wait_or_run_seconds':time.time()-start,'status_inferred_from_complete_artifact':bool(pid)}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(one,'body01',646985),pool.submit(one,'body02',647780)]+[pool.submit(one,'body%02d'%i) for i in range(3,11)]
 for f in concurrent.futures.as_completed(futures):
  row=f.result()
  with (OUT/'D_execution.jsonl').open('a') as o:o.write(json.dumps(row)+'\n')
  print(row,flush=True)
