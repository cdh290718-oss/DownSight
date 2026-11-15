import concurrent.futures,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_auto_loss_comparison_20261115';python=json.loads((ROOT/'outputs/synthetic_original_20261108/configs/left.json').read_text())['paths']['body_python']
def one(pair,method):
 dest=OUT/pair/'auto'; points=json.loads((dest/'observations.json').read_text())['points']
 if method=='3d' and not any(p['accepted'] for p in points):return {'pair':pair,'method':method,'status':'no_observations','exit_status':None}
 env=os.environ.copy();env.update(D_BATCH_PAIR=pair,D_BATCH_CONDITION='auto',D_METHOD=method,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');start=time.time()
 with (dest/(method+'.log')).open('x') as log:p=subprocess.run([python,str(RUN/'fit.py'),'--case','D'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 return {'pair':pair,'method':method,'status':'ok' if p.returncode==0 else 'failed','exit_status':p.returncode,'elapsed_seconds':time.time()-start}
with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
 futures=[pool.submit(one,f'body{i:02d}',m) for i in range(1,5) for m in ['3d','mixed','2d']]
 with (OUT/'execution.jsonl').open('x') as f:
  for future in concurrent.futures.as_completed(futures):
   r=future.result();f.write(json.dumps(r)+'\n');f.flush();print(json.dumps(r),flush=True)
