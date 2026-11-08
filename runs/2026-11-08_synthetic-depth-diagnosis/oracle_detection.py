"""Diagnostic oracle: replace in-frame body target coordinates, keep scores and other targets."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
from egosmplx_pipeline.body_common import JOINT_MAPPING
R=Path(__file__).resolve().parents[2];D=R/'outputs/synthetic_depth_diagnosis_20261108';O=R/'outputs/synthetic_original_20261108';run=Path(__file__).resolve().parent
cfg=json.loads((O/'configs/left.json').read_text());f=O/'fits/left/frames/left';paths=cfg['paths']
obs=json.loads((O/'fits/left/observations/left/sapiens/keypoints.json').read_text()); names=json.loads((O/'dataset/joint_names_egosmplx65.json').read_text())['names']
selected={x['name'] for x in json.loads((f/'rigid/optimization_report.json').read_text())['selected']};changed=[]
with np.load(O/'dataset/samples/000001/left_egosmplx65.npz',allow_pickle=False) as gt:
    uv=gt['joints_2d_px'];inside=gt['in_frame_no_occlusion_test']
    for q in obs['keypoints']:
        n=q['name']
        if n in selected and n in JOINT_MAPPING:
            i=names.index(JOINT_MAPPING[n])
            if inside[i]:q['x'],q['y']=map(float,uv[i]);changed.append(n)
target=D/'oracle_keypoints.json';target.write_text(json.dumps(obs,indent=2)+'\n')
env=os.environ.copy();env.update(PYTHONPATH=str(R),EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo'],EGOSMPLX_HEAD_NECK=json.dumps(cfg['head_neck']),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(D/'runtime'),XDG_CACHE_HOME=str(D/'runtime/cache'),MPLCONFIGDIR=str(D/'runtime/matplotlib'))
env.pop('EGOSMPLX_V2',None);env.pop('EGOSMPLX_V2_HAND_STAGE',None)
cmd=[paths['body_python'],str(run/'ablate.py'),'--scope','all','--repo-root',paths['egosmplx_repo'],'--source-npz',str(f/'rigid/optimized_params.npz'),'--image',str(O/'dataset/samples/000001/left.png'),'--calibration',str(O/'dataset/calibration/cam3_left_lower.egosmplx.json'),'--automatic-keypoints',str(target),'--output-dir',str(D/'oracle'),'--steps',str(cfg['fitting']['pose_steps'])]
t=time.time()
with (D/'oracle.log').open('x') as log:r=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
(run/'evidence/oracle_execution.json').write_text(json.dumps({'exit_status':r.returncode,'elapsed_seconds':time.time()-t,'changed_points':changed,'scope':'GT substituted solely for in-frame selected body targets in body pose stage; not an inference result; all other targets, rigid input, weights, and constraints unchanged'},indent=2)+'\n');print('oracle',r.returncode,flush=True)
