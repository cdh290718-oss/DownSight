"""Run independent cases with at most two CPU workers; preserve failure receipts."""
import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/initialization_ablation_20261109'
cfg=json.loads((ROOT/'outputs/synthetic_original_20261108/configs/left.json').read_text());python=cfg['paths']['body_python']
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(OUT/'runtime'),XDG_CACHE_HOME=str(OUT/'runtime/cache'),MPLCONFIGDIR=str(OUT/'runtime/matplotlib'))
for k in ['EGOSMPLX_V2','EGOSMPLX_V2_HAND_STAGE','EGOSMPLX_HEAD_NECK']:env.pop(k,None)
def run(case,seed):
 name=case+'_'+seed;cmd=[python,str(RUN/'fit.py'),'--case',case];start=time.time()
 with (OUT/(name+'.log')).open('x') as f:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 row=dict(case=case,seed=seed,exit_status=p.returncode,elapsed_seconds=time.time()-start,command=['<BODY_PYTHON>',str((RUN/'fit.py').relative_to(ROOT)),'--case',case],source_sha256={n:hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in ['fit.py','common.py','initialization.py','settings.json','cases.json']})
 if p.returncode:
  text=(OUT/(name+'.log')).read_text().replace(str(ROOT.parent),'<WORKSPACE>').replace('/usr/','<SYSTEM>/');(RUN/'evidence'/(name+'_failure.txt')).write_text(text)
 return row
cases=[(c['case'],c['seed']) for c in json.loads((RUN/'cases.json').read_text())]
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(run,c,s) for c,s in cases]
 for f in concurrent.futures.as_completed(futures):
  row=f.result();records.append(row);(RUN/'evidence/execution.json').write_text(json.dumps(records,indent=2)+'\n');print(row['case'],row['seed'],row['exit_status'],flush=True)
if any(q['exit_status'] for q in records):raise SystemExit(1)
