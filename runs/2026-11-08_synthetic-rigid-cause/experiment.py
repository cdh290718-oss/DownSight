"""Controlled rigid ablations; oracle shape, pose and selected image targets."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import load,read
from egosmplx_pipeline.body_common import JOINT_MAPPING
R=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;O=R/'outputs/synthetic_original_20261108';D=R/'outputs/synthetic_rigid_cause_20261108'; S=O/'dataset/samples/000001';f=O/'fits/left/frames/left';cfg=read(O/'configs/left.json');paths=cfg['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,Camera=setup(paths['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model)
raw=load(f/'raw/params.npz'); gt=load(S/'smplx_reconstruction.npz');obs=read(O/'fits/left/observations/left/sapiens/keypoints.json');names=read(O/'dataset/joint_names_egosmplx65.json')['names'];selected={x['name'] for x in read(f/'rigid/optimization_report.json')['selected']}
with np.load(S/'left_egosmplx65.npz',allow_pickle=False) as g:uv=g['joints_2d_px'].copy();inside=g['in_frame_no_occlusion_test'].copy()
for shape in ['pred_shape','gt_shape']:
 for pose in ['pred_pose','gt_pose']:
    p={k:v.copy() for k,v in raw.items()}
    if shape=='gt_shape':p['betas']=gt['betas'].reshape(p['betas'].shape)
    if pose=='gt_pose':p['body_pose']=gt['body_pose'].reshape(p['body_pose'].shape)
    mesh,joints=fusion.geometry(p);p['smplx_mesh_cam']=mesh.astype(np.float32);p['smplx_body_joints_3d_local']=(joints[:25]-p['transl'].reshape(3)).astype(np.float32)
    np.savez_compressed(D/(shape+'_'+pose+'.npz'),**p)
for target in ['detector_all','detector_inframe','oracle_inframe']:
    data=json.loads(json.dumps(obs));changed=[]
    for q in data['keypoints']:
        n=q['name']
        if n not in selected or n not in JOINT_MAPPING:continue
        i=names.index(JOINT_MAPPING[n])
        if target!='detector_all' and not inside[i]:q['score']=0.;changed.append(n)
        elif target=='oracle_inframe' and inside[i]:q['x'],q['y']=map(float,uv[i]);changed.append(n)
    (D/(target+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    (D/(target+'_changed.json')).write_text(json.dumps(changed)+'\n')
env=os.environ.copy();env.update(PYTHONPATH=str(R),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(D/'runtime'),XDG_CACHE_HOME=str(D/'runtime/cache'),MPLCONFIGDIR=str(D/'runtime/matplotlib'));env.pop('EGOSMPLX_V2',None)
results=[]
for shape,pose,target in [('pred_shape','pred_pose','detector_all'),('gt_shape','pred_pose','detector_all'),('pred_shape','gt_pose','detector_all'),('gt_shape','gt_pose','detector_all'),('pred_shape','pred_pose','detector_inframe'),('gt_shape','pred_pose','detector_inframe'),('pred_shape','gt_pose','detector_inframe'),('gt_shape','gt_pose','detector_inframe'),('pred_shape','pred_pose','oracle_inframe'),('gt_shape','pred_pose','oracle_inframe'),('pred_shape','gt_pose','oracle_inframe'),('gt_shape','gt_pose','oracle_inframe')]:
    name=shape+'_'+pose+'_'+target;out=D/name
    cmd=[paths['body_python'],'-m','egosmplx_pipeline.body_rigid','--repo-root',paths['egosmplx_repo'],'--source-npz',str(D/(shape+'_'+pose+'.npz')),'--image',str(S/'left.png'),'--calibration',str(O/'dataset/calibration/cam3_left_lower.egosmplx.json'),'--keypoints',str(D/(target+'.json')),'--output-dir',str(out),'--steps',str(cfg['fitting']['rigid_steps']),'--device','cpu']
    t=time.time()
    with (D/(name+'.log')).open('x') as log:r=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
    row={'name':name,'shape':shape,'pose':pose,'target':target,'exit_status':r.returncode,'elapsed_seconds':time.time()-t};results.append(row);(RUN/'evidence/execution.json').write_text(json.dumps(results,indent=2)+'\n');print(name,r.returncode,flush=True)
