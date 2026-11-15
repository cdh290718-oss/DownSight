import json,os,sys,importlib.util
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/gt2d_vs_sapiens_20261115';ARCH=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';read=lambda p:json.loads(Path(p).read_text());paths=read(OUT/'body01_auto/model_paths.json');os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo'])
from egosmplx_pipeline.bootstrap import setup
model,Camera=setup(paths['egosmplx_repo']);from egosmplx_pipeline import fusion
fusion.initialize(model);sys.path.insert(0,str(ARCH));from evaluation_common import align
records=[json.loads(line) for line in (OUT/'execution_retry.jsonl').read_text().splitlines()];assert len(records)==6 and len({(r['pair'],r['mode']) for r in records})==6
prep=read(RUN/'evidence/preparation.json');ref=read(ROOT/'runs/2026-11-15_unrealego-gt2d/evidence/preparation_ideal.json');R=np.array(ref['rig_rotation']);t=np.array(ref['rig_translation_m']);gt=np.load(ARCH/'body_gt_official.npz');ids=[7,9,11,13,8,10,12,2,4,6,19,1,3,5,16];names=ref['joint_names'];spec=importlib.util.spec_from_file_location('rasterizer',ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py');raster=importlib.util.module_from_spec(spec);spec.loader.exec_module(raster);faces=np.asarray(model.face,np.int32);dominant=model.layer['neutral'].lbs_weights.detach().numpy().argmax(1);head=(dominant==model.orig_joints_name.index('Head')).astype(np.float32);cams={s:Camera(ARCH/'body_calibration_fisheye'/('camera_'+s+'.json')) for s in ['left','right']};rays={}
for side,c in cams.items():
 Y,X=np.indices((720,1280));rays[side]=c.camera2world_ray(np.stack([X.ravel(),Y.ravel()],axis=1)).reshape(720,1280,3).astype(np.float32)
rows=[]
for rec in sorted(records,key=lambda r:(r['pair'],r['mode'])):
 dest=OUT/(rec['pair']+'_'+rec['mode']+'_retry');row=dict(rec)
 if rec['exit_status']!=0:row['status']='failed';rows.append(row);continue
 p=dict(np.load(dest/'D_left/final.npz'));j=p['smplx_joint_cam'];v=p['smplx_mesh_cam'];idx=(int(rec['pair'][-2:])-1)*2;target=gt['joints_m'][idx].astype(float);pred=j[ids].astype(float);absd=np.linalg.norm(pred-target,axis=1)*1000;pad=np.linalg.norm(align(pred,target)-target,axis=1)*1000;opt=read(dest/'D_left/optimization.json');row.update(status='ok',absolute15_mean_mm=float(absd.mean()),absolute15_per_joint_mm=dict(zip(names,absd.tolist())),pa15_mean_mm=float(pad.mean()),pa15_per_joint_mm=dict(zip(names,pad.tolist())),selected_step=opt['best_step'],shape_rank=opt['initialization']['retained_rank'],shape_basis=opt['initialization']['basis'],betas_final=opt['betas_final'],convergence=opt['convergence'],head_position_error_mm=opt['final']['head_position_error_mm'],final_sha256=opt['final']['sha256'],images=[])
 for i,side in enumerate(['left','right']):
  c=cams[side];vv=v.astype(float) if i==0 else v@R.T+t;im=cv2.imread(str(ARCH/'body_input'/(rec['pair']+'_'+side+'.png')));uv=c.world2camera(vv).astype(np.float32);depth=np.full((720,1280),np.inf,np.float32);colors=np.zeros_like(im);hp=np.zeros((720,1280),np.float32);drawn,rejected=raster.raster(vv.astype(np.float32),faces,head,uv,rays[side],depth,colors,hp,180.);mask=np.isfinite(depth);im[mask]=(.45*im[mask]+.55*colors[mask]).astype(np.uint8);file=rec['pair']+'-'+rec['mode']+'-'+side+'.jpg';cv2.imwrite(str(RUN/'assets'/file),im,[cv2.IMWRITE_JPEG_QUALITY,90]);row['images'].append({'camera':side,'file':file,'rejected_faces':int(rejected),'visible_pixels':int(mask.sum())})
 rows.append(row);print(rec['pair'],rec['mode'],row['pa15_mean_mm'],flush=True)
for item in prep['pairs']:
 pair=item['pair'];a=[r for r in rows if r['pair']==pair and r['status']=='ok']
 if len(a)==2:
  x,y=[r['shape_basis'] for r in a];assert np.allclose(x,y),pair
 if item['matched']==0:rows.append({'pair':pair,'status':'not_run','reason':'no common usable targets; both methods cannot be compared','matched_target_count':0})
q={'note':'GT15 only evaluated after fitting; Head 3D and calibration are annotation-derived; 2D GT control and auto use same accepted indices and covariance; 1px assumed detector sigma, not measured','records':rows,'command':'<BODY_PYTHON> runs/2026-11-15_gt2d-vs-sapiens/evaluate.py','exit_status':0};(RUN/'evidence/results.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
