"""Ablate selected detector groups without using truth point coordinates."""
import json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;O=R/'outputs/synthetic_original_20261108';D=R/'outputs/synthetic_rigid_cause_20261108';S=O/'dataset/samples/000001';cfg=json.loads((O/'configs/left.json').read_text());paths=cfg['paths'];env=os.environ.copy();env.update(PYTHONPATH=str(R),EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(D/'runtime'),XDG_CACHE_HOME=str(D/'runtime/cache'),MPLCONFIGDIR=str(D/'runtime/matplotlib'));env.pop('EGOSMPLX_V2',None)
obs=json.loads((D/'detector_all.json').read_text());results=[]
for label,omit in [('no_nose',['nose']),('no_shoulders',['left_shoulder','right_shoulder']),('no_wrists',['left_wrist','right_wrist'])]:
    data=json.loads(json.dumps(obs))
    for q in data['keypoints']:
        if q['name'] in omit:q['score']=0.
    target=D/(label+'.json');target.write_text(json.dumps(data,indent=2)+'\n');name='pred_shape_pred_pose_'+label
    cmd=[paths['body_python'],'-m','egosmplx_pipeline.body_rigid','--repo-root',paths['egosmplx_repo'],'--source-npz',str(D/'pred_shape_pred_pose.npz'),'--image',str(S/'left.png'),'--calibration',str(O/'dataset/calibration/cam3_left_lower.egosmplx.json'),'--keypoints',str(target),'--output-dir',str(D/name),'--steps',str(cfg['fitting']['rigid_steps']),'--device','cpu'];t=time.time()
    with (D/(name+'.log')).open('x') as log:r=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
    row={'name':name,'shape':'pred_shape','pose':'pred_pose','target':label,'omitted':omit,'exit_status':r.returncode,'elapsed_seconds':time.time()-t};results.append(row);(RUN/'evidence/group_execution.json').write_text(json.dumps(results,indent=2)+'\n');print(label,r.returncode,flush=True)
