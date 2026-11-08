"""Execute independent SE3-only and SE3+contour joint controls on both views."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/synthetic_head_se3_contour_20261108';runtime=OUT/'runtime';runtime.mkdir(parents=True,exist_ok=True);(RUN/'evidence').mkdir(exist_ok=True)
cfg=json.loads((ROOT/'outputs/synthetic_original_20261108/configs/left.json').read_text());python=cfg['paths']['body_python']
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(runtime),XDG_CACHE_HOME=str(runtime/'cache'),MPLCONFIGDIR=str(runtime/'matplotlib'))
for k in ['EGOSMPLX_V2','EGOSMPLX_V2_HAND_STAGE','EGOSMPLX_HEAD_NECK']:env.pop(k,None)
records=[]
for side in ['left','right']:
 for flag in [0,1]:
    name=side+('_se3_contour_joint' if flag else '_se3_joint');cmd=[python,str(RUN/'fit.py'),'--side',side,'--contour',str(flag)];start=time.time()
    with (OUT/(name+'.log')).open('x') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    row=dict(side=side,contour=bool(flag),exit_status=r.returncode,elapsed_seconds=time.time()-start,command=['<BODY_PYTHON>','runs/2026-11-08_synthetic-head-se3-contour/fit.py','--side',side,'--contour',str(flag)],code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'fit.py',RUN/'contour.py',RUN/'common.py',RUN/'settings.json']});records.append(row)
    (RUN/'evidence/execution.json').write_text(json.dumps(records,indent=2)+'\n');print(name,r.returncode,flush=True)
    if r.returncode:
        s=(OUT/(name+'.log')).read_text(errors='replace').replace(str(ROOT.parent),'<WORKSPACE>').replace('/usr/','<SYSTEM>/');(RUN/'evidence'/(name+'_failure.txt')).write_text(s.rstrip()+'\n')
