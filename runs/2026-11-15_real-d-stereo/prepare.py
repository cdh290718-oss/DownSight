import json,sys,os
from pathlib import Path
import numpy as np, torch, cv2
ROOT=Path(__file__).resolve().parents[2]; RUN=Path(__file__).resolve().parent; OUT=ROOT/'outputs/real_d_stereo_20261115'
sys.path.insert(0,str(ROOT/'runs/2026-11-04_session-hand6-v2/code'))
from prepare import triangulate, Camera
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.body_common import JOINT_MAPPING
from common import head_rotation_numpy
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
data=read(ROOT/'outputs/session_hand6_v2_no_margin_rerun_20261104/prepared/inputs.json'); rec=[r for r in data['records'] if r['pair']=='new_001']; assert [r['camera'] for r in rec]==['cam3','cam4']
paths=read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json'); os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model); cams=[Camera(r['row']['calibration']) for r in rec]; A=np.array(read(data['calibration'])['poseCamera2']['A']);R=A[:3,:3]; t=A[:3,3]/1000
views={'left':{'camera':{'left_to_view':np.eye(4).tolist()}},'right':{'camera':{'left_to_view':np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist()}}};write(OUT/'inputs.json',{'views':views});write(OUT/'model_paths.json',paths)
look=[{p['name']:p for p in read(r['row']['predictions']['sapiens'])['keypoints']} for r in rec];rows=[]
for name in ['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']:
 p=[x[name] for x in look]; q=triangulate(cams,R,t,[[x['x'],x['y']] for x in p]); kind=name.split('_')[-1]; sigma={'shoulder':.03,'hip':.03,'elbow':.02,'knee':.02,'ankle':.015,'wrist':.008}[kind]; row=dict(name=name,index=model.joints_name.index(JOINT_MAPPING[name]),group='shoulder' if kind=='shoulder' else 'wrist' if kind=='wrist' else 'body',source='sapiens',accepted=q['accepted'] and all(x['score']>=.3 for x in p),triangulation=q)
 if q['accepted']: row.update(xyz=q['point_cam3_m'],covariance=(np.array(q['covariance_cam3_m2'])+np.eye(3)*sigma**2).tolist())
 rows.append(row)
ctx=read(rec[0]['context']); pts=np.array(ctx['tri']); W=np.array(ctx['whiten']); weights=np.array(ctx['tri_weights'])
for i,side in enumerate(['left','right']):
 for k in [0,5,9,13,17]:
  ok=bool(weights[i,k]>0); row=dict(name=side+'_hand_'+str(k),index=([12+i]+list(range(25+20*i,45+20*i)))[k],group='hand_'+side,source='wilor',accepted=ok)
  if ok: row.update(xyz=pts[i,k].tolist(),covariance=(np.linalg.inv(W[i,k].T@W[i,k])+np.eye(3)*.008**2).tolist())
  rows.append(row)
write(OUT/'triangulated.json',{'points':rows});write(RUN/'evidence/triangulated.json',{'points':rows,'note':'Fisheye triangulation prior run uses 8 px and 8 mm floor; added D definition noise. Excludes calibration and sync uncertainty.'})
p=dict(np.load(rec[0]['original']['body']));layer=model.layer['neutral'].cpu().eval()
with torch.no_grad(): native=layer(**{k:torch.as_tensor(p[k],dtype=torch.float32).reshape(1,-1) for k in fusion.PARAMS}).joints[0].numpy()+p['transl'].reshape(3)
anchor={'position':native[15].tolist(),'rotation':head_rotation_numpy(p).tolist()}; (OUT/'anchors').mkdir(exist_ok=True);write(OUT/'anchors/oracle.json',anchor)
write(RUN/'evidence/anchor.json',{'source':'cam3 historical original fused body parameters; estimated, not ground truth','anchor':anchor,'accepted_targets':sum(x['accepted'] for x in rows),'targets':len(rows)})
print('accepted',sum(x['accepted'] for x in rows),'of',len(rows))
