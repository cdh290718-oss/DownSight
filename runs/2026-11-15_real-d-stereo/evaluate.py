import json,os,sys,importlib.util,shutil
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/real_d_stereo_20261115';read=lambda p:json.loads(Path(p).read_text())
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.body_common import JOINT_MAPPING
paths=read(OUT/'model_paths.json');os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,Camera=setup(paths['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model)
spec=importlib.util.spec_from_file_location('rasterizer',ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py'); raster=importlib.util.module_from_spec(spec);spec.loader.exec_module(raster)
data=read(ROOT/'outputs/session_hand6_v2_no_margin_rerun_20261104/prepared/inputs.json');rec=[r for r in data['records'] if r['pair']=='new_001'];A=np.array(read(data['calibration'])['poseCamera2']['A']);R=A[:3,:3];t=A[:3,3]/1000
names=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'];ids=[model.joints_name.index(JOINT_MAPPING[n]) for n in names]; faces=np.asarray(model.face,dtype=np.int32); dominant=model.layer['neutral'].lbs_weights.detach().numpy().argmax(1);head=(dominant==model.orig_joints_name.index('Head')).astype(np.float32)
results=[];(RUN/'assets').mkdir(exist_ok=True)
for r in rec:
 cam=Camera(r['row']['calibration']);im=cv2.imread(r['row']['image']);lookup={p['name']:p for p in read(r['row']['predictions']['sapiens'])['keypoints']};goal=np.array([[lookup[n]['x'],lookup[n]['y']] for n in names]); yy,xx=np.indices(im.shape[:2]);rays=cam.camera2world_ray(np.stack([xx.ravel(),yy.ravel()],axis=1)).reshape(*im.shape[:2],3).astype(np.float32)
 for case in ['D','plus','minus','original','v2']:
  if case in ['original','v2']:
   path=Path(r['original']['body']) if case=='original' else Path(read(r['config'])['output'])/'frames'/r['id']/'fused/body_params.npz'; p=dict(np.load(path));_,j=fusion.geometry(p)
  else:
   z=np.load(OUT/(case+'_left')/'final.npz');v=z['smplx_mesh_cam'].astype(float);j=z['smplx_joint_cam'].astype(float)
   if r['camera']=='cam4':v=v@R.T+t;j=j@R.T+t
   uv=cam.world2camera(v).astype(np.float32);depth=np.full(im.shape[:2],np.inf,np.float32);colors=np.zeros_like(im);hp=np.zeros(im.shape[:2],np.float32);drawn,rejected=raster.raster(v.astype(np.float32),faces,head,uv,rays,depth,colors,hp,180.);visible=np.isfinite(depth);overlay=im.copy();overlay[visible]=(.45*im[visible]+.55*colors[visible]).astype(np.uint8)
   cv2.imwrite(str(RUN/'assets'/(r['camera']+'-'+case+'.jpg')),overlay,[cv2.IMWRITE_JPEG_QUALITY,93])
  pred=cam.world2camera(j[ids].astype(float));error=np.linalg.norm(pred-goal,axis=1);results.append(dict(camera=r['camera'],case=case,mean_px=float(error.mean()),median_px=float(np.median(error)),per_point=dict(zip(names,error.tolist()))))
 for case in ['original','v2']:
  shutil.copyfile(ROOT/'outputs/session_hand6_v2_no_margin_rerun_20261104/evaluation_corrected'/('new_001-'+r['camera']+'-'+case+'.jpg'),RUN/'assets'/(r['camera']+'-'+case+'.jpg'))
(RUN/'evidence/metrics.json').write_text(json.dumps({'description':'12 common automatic Sapiens targets; 2D pixel distance is agreement, not ground truth. Historical v2 is separate per-view fitting with MANO replacement.','frames':results},indent=2)+'\n')
print(json.dumps(results,indent=2))
