import json,sys,os,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_d_pilot_20261115';BASE=ROOT.parent
sys.path.insert(0,str(ROOT/'runs/2026-11-04_session-hand6-v2/code'))
from prepare import triangulate,Camera
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.body_common import JOINT_MAPPING
from common import head_rotation_numpy
read=lambda p:json.loads(Path(p).read_text());write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
archive=BASE/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';sample=read(archive/'body_samples.json')['samples'];datafile=BASE/'EgoSMPLX/data/UnrealEgo_RW_Test/prepared/samples.npz';data=np.load(datafile);ia,ib=[x['sample_index'] for x in sample[2:4]]
# Estimate rig transform from fixed calibration pair 02; never read body01 GT15 during fitting preparation.
x=data['keypoints3d'][ia].astype(float);y=data['keypoints3d'][ib].astype(float);xc=x.mean(0);yc=y.mean(0);u,s,vt=np.linalg.svd((x-xc).T@(y-yc));R=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T;t=yc-R@xc;errs=np.linalg.norm(x@R.T+t-y,axis=1)
paths=read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json');os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model);cams=[Camera(archive/'body_calibration_fisheye'/('camera_'+side+'.json')) for side in ['left','right']]; rec=sample[:2];look=[]
for a in rec:
 path=archive/'body_work_full_image'/a['id']/'observations'/a['id']/'sapiens/keypoints.json';look.append({q['name']:q for q in read(path)['keypoints']})
rows=[]
for name in ['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']:
 if not all(name in x for x in look):continue
 p=[x[name] for x in look];xy=np.array([[z['x'],z['y']] for z in p]);q=triangulate(cams,R,t,xy);kind=name.split('_')[-1];sig={'shoulder':.03,'hip':.03,'elbow':.02,'knee':.02,'ankle':.015,'wrist':.008}[kind]; ok=q['accepted'] and all(z['score']>=.3 for z in p) and bool(((xy[:,0]>=280)&(xy[:,0]<1000)&(xy[:,1]>=0)&(xy[:,1]<720)).all());row=dict(name=name,index=model.joints_name.index(JOINT_MAPPING[name]),group='shoulder' if kind=='shoulder' else 'wrist' if kind=='wrist' else 'body',source='sapiens',accepted=ok,triangulation=q,scores=[z['score'] for z in p],xy=xy.tolist())
 if q['accepted']:row.update(xyz=q['point_cam3_m'],covariance=(np.array(q['covariance_cam3_m2'])+np.eye(3)*sig**2).tolist())
 rows.append(row)
views={'left':{'camera':{'left_to_view':np.eye(4).tolist()}},'right':{'camera':{'left_to_view':np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist()}}};write(OUT/'inputs.json',{'views':views});write(OUT/'model_paths.json',paths);write(OUT/'triangulated.json',{'points':rows})
p=dict(np.load(archive/'body_work_full_image/body01_left/frames/body01_left/fused/body_params.npz'));anchor={'position':data['keypoints3d_fit'][rec[0]['sample_index'],0].astype(float).tolist(),'rotation':head_rotation_numpy(p).tolist()};(OUT/'anchors').mkdir(exist_ok=True);write(OUT/'anchors/oracle.json',anchor)
write(RUN/'evidence/preparation.json',{'exit_status':0,'sample':'body01','head_position':'official Head 3D annotation, left camera, metres; semantic agreement with SMPL-X Head not validated','head_orientation':'historical body01 left fused model motion chain; network-derived, not GT','head':anchor,'rig':'rigid transform estimated from body02 paired official GT15, introduces benchmark annotation dependence','rig_rotation':R.tolist(),'rig_translation_m':t.tolist(),'calibration_pair_fit_residual_m':errs.tolist(),'observations':'automatic Sapiens only; no hand 3D targets for pilot','targets':rows,'prepared_dataset_sha256':hashlib.sha256(datafile.read_bytes()).hexdigest()});print('accepted',sum(x['accepted'] for x in rows),'of',len(rows))
