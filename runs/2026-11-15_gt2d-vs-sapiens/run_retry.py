import os,json,subprocess,concurrent.futures,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/gt2d_vs_sapiens_20261115';python=json.loads((ROOT/'outputs/synthetic_original_20261108/configs/left.json').read_text())['paths']['body_python'];prep=json.loads((RUN/'evidence/preparation.json').read_text())
def one(pair,mode):
 name=pair+'_'+mode+'_retry';dest=OUT/name;env=os.environ.copy();env.update(D_BATCH_PAIR=name,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(dest/'runtime'),XDG_CACHE_HOME=str(dest/'runtime/cache'),MPLCONFIGDIR=str(dest/'runtime/matplotlib'));cmd=[python,str(RUN/'fit.py'),'--case','D'];start=time.time()
 with (dest/'D.log').open('x') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 return {'pair':pair,'mode':mode,'exit_status':r.returncode,'elapsed_seconds':time.time()-start}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(one,r['pair'],mode) for r in prep['pairs'] if r['matched']>0 for mode in ['auto','official']]
 for f in concurrent.futures.as_completed(futures):
  row=f.result()
  with (OUT/'execution_retry.jsonl').open('a') as o:o.write(json.dumps(row)+'\n')
  print(row,flush=True)
