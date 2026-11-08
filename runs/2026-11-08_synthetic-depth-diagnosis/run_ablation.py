"""Run paired cam3 pose-stage fits with identical saved observations and init."""
import json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[2]; D=R/'outputs/synthetic_depth_diagnosis_20261108'; O=R/'outputs/synthetic_original_20261108'; run=Path(__file__).resolve().parent
cfg=json.loads((O/'configs/left.json').read_text());paths=cfg['paths'];frame=O/'fits/left/frames/left'
env=os.environ.copy();env.update(PYTHONPATH=str(R),EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo'],EGOSMPLX_HEAD_NECK=json.dumps(cfg['head_neck']),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(D/'runtime'),XDG_CACHE_HOME=str(D/'runtime/cache'),MPLCONFIGDIR=str(D/'runtime/matplotlib'))
env.pop('EGOSMPLX_V2',None);env.pop('EGOSMPLX_V2_HAND_STAGE',None)
records=[]
for scope in ['all','non_head_neck']:
    dest=D/scope
    cmd=[paths['body_python'],str(run/'ablate.py'),'--scope',scope,'--repo-root',paths['egosmplx_repo'],'--source-npz',str(frame/'rigid/optimized_params.npz'),'--image',str(O/'dataset/samples/000001/left.png'),'--calibration',str(O/'dataset/calibration/cam3_left_lower.egosmplx.json'),'--automatic-keypoints',str(O/'fits/left/observations/left/sapiens/keypoints.json'),'--output-dir',str(dest),'--steps',str(cfg['fitting']['pose_steps'])]
    t=time.time()
    with (D/(scope+'.log')).open('x') as f:r=subprocess.run(cmd,env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT)
    records.append({'scope':scope,'exit_status':r.returncode,'elapsed_seconds':time.time()-t,'steps':cfg['fitting']['pose_steps']})
    (run/'evidence/ablation_execution.json').write_text(json.dumps(records,indent=2)+'\n')
    print(scope,r.returncode,flush=True)
