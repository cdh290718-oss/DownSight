"""Run official noisy observations with paired losses, reusing exactly identical branches."""
import concurrent.futures,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_official_noise_v2_20261116';python=json.loads((ROOT/'outputs/synthetic_original_20261108/configs/left.json').read_text())['paths']['body_python'];items=json.loads((RUN/'evidence/preparation.json').read_text())['conditions']
def one(item,method):
 pair,condition=item['pair'],item['condition'];dest=OUT/pair/condition;env=os.environ.copy();env.update(D_BATCH_PAIR=pair,D_BATCH_CONDITION=condition,D_METHOD=method,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');start=time.time()
 with (dest/(method+'.log')).open('x') as log:r=subprocess.run([python,str(RUN/'fit.py'),'--case','D'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 return {'pair':pair,'condition':condition,'method':method,'status':'ok' if r.returncode==0 else 'failed','exit_status':r.returncode,'elapsed_seconds':time.time()-start}
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
 jobs={}
 for item in items:
  for m in ['3d','mixed','2d']:
   if m=='mixed' and item['2d_fallback']==0:continue
   jobs[pool.submit(one,item,m)]=item
 with (OUT/'execution.jsonl').open('x') as f:
  for future in concurrent.futures.as_completed(jobs):
   r=future.result();f.write(json.dumps(r)+'\n');f.flush();print(json.dumps(r),flush=True)
   if r['method']=='3d' and jobs[future]['2d_fallback']==0:
    alias=dict(r,method='mixed',reused_from='3d',reason='All valid observations use identical 3D branch; no 2D fallback',elapsed_seconds=0);f.write(json.dumps(alias)+'\n');f.flush()
