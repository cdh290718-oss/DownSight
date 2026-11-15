import json,os,sys,hashlib,importlib.util
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_gt16_20261115';ARCH=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';read=lambda p:json.loads(Path(p).read_text());paths=read(OUT/'body01/model_paths.json');os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo'])
from egosmplx_pipeline.bootstrap import setup
model,Camera=setup(paths['egosmplx_repo']);from egosmplx_pipeline import fusion
fusion.initialize(model);sys.path.insert(0,str(ARCH));from evaluation_common import align
records=[json.loads(line) for line in (OUT/'D_execution.jsonl').read_text().splitlines()];assert len(records)==4 and len({(r['pair'],r['method']) for r in records})==4
spec=importlib.util.spec_from_file_location('rasterizer',ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py');raster=importlib.util.module_from_spec(spec);spec.loader.exec_module(raster);prep=read(RUN/'evidence/preparation.json');R=np.array(prep['rig_rotation']);t=np.array(prep['rig_translation_m']);faces=np.asarray(model.face,np.int32);dominant=model.layer['neutral'].lbs_weights.detach().numpy().argmax(1);head=(dominant==model.orig_joints_name.index('Head')).astype(np.float32);assets=RUN/'assets';assets.mkdir(exist_ok=True);cams={s:Camera(ARCH/'body_calibration_fisheye'/('camera_'+s+'.json')) for s in ['left','right']};rays={}
for side,c in cams.items():
 Y,X=np.indices((720,1280));rays[side]=c.camera2world_ray(np.stack([X.ravel(),Y.ravel()],axis=1)).reshape(720,1280,3).astype(np.float32)
rows=[]
for rec in sorted(records,key=lambda r:(r['pair'],r['method'])):
 dest=OUT/rec['pair'];row=dict(rec)
 if rec['exit_status']!=0:
  row['status']='failed';failure=dest/('D_left' if rec['method']=='D' else 'original_left')/'failure.json'
  if failure.exists():row['failure']=read(failure)['reason']
  rows.append(row);continue
 p=dict(np.load(dest/('D_left' if rec['method']=='D' else 'original_left')/'final.npz'));j=p['smplx_joint_cam'];v=p['smplx_mesh_cam'];target=read(dest/'gt16.json');ids=target['model_indices'][1:];xyz=np.array(target['xyz']);pred=j[ids].astype(float)
 with torch.no_grad():native=model.layer['neutral'](**{k:torch.as_tensor(p[k],dtype=torch.float32).reshape(1,-1) for k in fusion.PARAMS}).joints[0].numpy()+p['transl'].reshape(3)
 absd=np.linalg.norm(pred-xyz[1:],axis=1)*1000;pad=np.linalg.norm(align(pred,xyz[1:])-xyz[1:],axis=1)*1000;headerr=float(np.linalg.norm(native[15]-xyz[0])*1000);opt=read(dest/('D_left' if rec['method']=='D' else 'original_left')/'optimization.json');row.update(status='ok',absolute15_mean_mm=float(absd.mean()),absolute15_per_joint_mm=absd.tolist(),pa15_mean_mm=float(pad.mean()),pa15_per_joint_mm=pad.tolist(),head_absolute_error_mm=headerr,selected_step=opt['best_step'],reload_mesh_max_m=opt['final']['reload_mesh_max_m'] if rec['method']=='D' else opt['reload_mesh_max_m'],convergence=opt.get('convergence'),final_sha256=opt['final']['sha256'] if rec['method']=='D' else opt['sha256']);
 for i,side in enumerate(['left','right']):
  c=cams[side];vv=v.astype(float) if i==0 else v@R.T+t;im=cv2.imread(str(ARCH/'body_input'/(rec['pair']+'_'+side+'.png')));uv=c.world2camera(vv).astype(np.float32);depth=np.full((720,1280),np.inf,np.float32);colors=np.zeros_like(im);hp=np.zeros((720,1280),np.float32);drawn,rejected=raster.raster(vv.astype(np.float32),faces,head,uv,rays[side],depth,colors,hp,180.);mask=np.isfinite(depth);im[mask]=(.45*im[mask]+.55*colors[mask]).astype(np.uint8);cv2.imwrite(str(assets/(rec['pair']+'-'+rec['method']+'-'+side+'.jpg')),im,[cv2.IMWRITE_JPEG_QUALITY,90])
 rows.append(row);print(rec['pair'],rec['method'],row['pa15_mean_mm'],flush=True)
out={'note':'GT16 used during fitting; same-point residuals only; not independent accuracy','records':rows,'expected_runs':4,'execution_records':len(records),'command':'<BODY_PYTHON> runs/2026-11-15_unrealego-gt16/evaluate_four.py','exit_status':0};(RUN/'evidence/results_four.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
