import os,sys,json,subprocess,concurrent.futures,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];run=Path(__file__).resolve().parent;out=root/'outputs/real_d_stereo_20261115';paths=json.loads((root/'outputs/synthetic_original_20261108/configs/left.json').read_text())['paths'];env=os.environ.copy();env.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(out/'runtime'),XDG_CACHE_HOME=str(out/'runtime/cache'),MPLCONFIGDIR=str(out/'runtime/matplotlib'))
def one(case):
 start=time.time()
 with (out/(case+'.log')).open('x') as f:p=subprocess.run([paths['body_python'],str(run/'fit.py'),'--case',case],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 return {'case':case,'exit_status':p.returncode,'elapsed_seconds':time.time()-start}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for result in pool.map(one,['D','plus','minus']):
  with (out/'execution.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
  print(result,flush=True)
