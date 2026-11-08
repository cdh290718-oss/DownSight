"""Re-run rigid with only depth penalty scope changed, same input and targets."""
import json,os,runpy,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'outputs/synthetic_depth_diagnosis_20261108';O=R/'outputs/synthetic_original_20261108';run=Path(__file__).resolve().parent
if len(sys.argv)>1:
    scope=sys.argv[1];from egosmplx_pipeline import v2_policy
    if scope=='non_head_neck':
        def min_z(v):return v[v2_policy.vertex_indices()[0],2].min()
        v2_policy.min_z=min_z
    elif scope=='none':v2_policy.min_z=lambda v:v[:,2].min()*0+1
    sys.argv=['egosmplx_pipeline.body_rigid']+sys.argv[2:];runpy.run_module('egosmplx_pipeline.body_rigid',run_name='__main__')
else:
    cfg=json.loads((O/'configs/left.json').read_text());paths=cfg['paths'];f=O/'fits/left/frames/left'
    env=os.environ.copy();env.update(PYTHONPATH=str(R),EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(D/'runtime'),XDG_CACHE_HOME=str(D/'runtime/cache'),MPLCONFIGDIR=str(D/'runtime/matplotlib'));env.pop('EGOSMPLX_V2',None)
    records=[]
    for scope in ['all','non_head_neck','none']:
        cmd=[paths['body_python'],str(run/'rigid_ablation.py'),scope,'--repo-root',paths['egosmplx_repo'],'--source-npz',str(f/'raw/params.npz'),'--image',str(O/'dataset/samples/000001/left.png'),'--calibration',str(O/'dataset/calibration/cam3_left_lower.egosmplx.json'),'--keypoints',str(O/'fits/left/observations/left/sapiens/keypoints.json'),'--output-dir',str(D/('rigid_'+scope)),'--steps',str(cfg['fitting']['rigid_steps']),'--device','cpu']
        t=time.time()
        with (D/('rigid_'+scope+'.log')).open('x') as log:r=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
        records.append({'scope':scope,'exit_status':r.returncode,'elapsed_seconds':time.time()-t});(run/'evidence/rigid_ablation_execution.json').write_text(json.dumps(records,indent=2)+'\n');print(scope,r.returncode,flush=True)
