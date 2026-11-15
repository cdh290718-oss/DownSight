import os,json,subprocess,concurrent.futures,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_gt2d_ideal_20261115';python=json.loads((ROOT/'outputs/synthetic_original_20261108/configs/left.json').read_text())['paths']['body_python']
def one(pair):
 dest=OUT/pair;env=os.environ.copy();env.update(D_BATCH_PAIR=pair,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(dest/'runtime'),XDG_CACHE_HOME=str(dest/'runtime/cache'),MPLCONFIGDIR=str(dest/'runtime/matplotlib'));cmd=[python,str(RUN/'fit.py'),'--case','D'];start=time.time()
 with (dest/'D.log').open('x') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 return {'pair':pair,'method':'D','exit_status':r.returncode,'elapsed_seconds':time.time()-start}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(one,'body%02d'%i) for i in range(1,5)]
 for f in concurrent.futures.as_completed(futures):
  row=f.result();
  with (OUT/'execution.jsonl').open('a') as o:o.write(json.dumps(row)+'\n')
  print(row,flush=True)
