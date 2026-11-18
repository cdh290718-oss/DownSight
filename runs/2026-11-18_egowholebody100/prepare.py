import pickle,json,sys,hashlib,random,concurrent.futures,os
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/egowholebody100_20261118';DATA=Path(os.environ['EWB_DATA_ROOT']);BACK=ROOT.parent/'EgoSMPLX';sys.path.insert(0,str(BACK/'main'))
from FishEyeCalibrated import FishEyeCameraCalibrated
cal=BACK/'main/fisheye.calibration_01_12.json';camera=FishEyeCameraCalibrated(str(cal));mapping=[('left_shoulder',5,'upperarm_l'),('right_shoulder',6,'upperarm_r'),('left_elbow',7,'lowerarm_l'),('right_elbow',8,'lowerarm_r'),('left_wrist',62,'hand_l'),('right_wrist',41,'hand_r'),('left_hip',9,'upperleg_l'),('right_hip',10,'upperleg_r'),('left_knee',11,'lowerleg_l'),('right_knee',12,'lowerleg_r'),('left_ankle',13,'foot_l'),('right_ankle',14,'foot_r')]
actions=sorted(x for x in DATA.iterdir() if x.is_dir());random.Random(20261118).shuffle(actions);selected=actions[:100];assert len(selected)==100

def one(arg):
 i,p=arg;j=pickle.load(open(p/'joint_info.pkl','rb'));c=pickle.load(open(p/'ego_camera_info.pkl','rb'));images=sorted((p/'img').glob('*.jpg'));assert len(j)==len(c)==len(images),(p.name,len(j),len(c),len(images));idx=len(images)//2;img=images[idx];im=cv2.imread(str(img));assert im is not None;h,w=im.shape[:2];assert (w,h)==tuple(camera.img_size)
 xyz=np.array([j[idx][name]['location'] for _,_,name in mapping]);xyz=(np.linalg.inv(c[idx])@np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]*[1,-1,-1];uv=np.array(camera.world2camera(xyz));points=[]
 for k,(name,sapiens,native) in enumerate(mapping):
  q=uv[k];valid=bool(np.isfinite(q).all() and 0<=q[0]<w and 0<=q[1]<h);points.append(dict(name=name,sapiens_index=sapiens,native_name=native,reference=q.tolist(),reference_in_image=valid))
 if i<4:
  overlay=im.copy()
  for k,pnt in enumerate(points):
   if pnt['reference_in_image']: pt=tuple(np.round(uv[k]).astype(int));cv2.circle(overlay,pt,6,(0,255,255),2);cv2.putText(overlay,str(k),pt,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
  cv2.imwrite(str(OUT/f'overlay_{i}.jpg'),overlay)
 return dict(id=f'ewb_{i:03d}',subject=DATA.name,action=p.name,frame_index=idx,image=str(img),width=w,height=h,image_sha256=hashlib.sha256(img.read_bytes()).hexdigest(),joint_sha256=hashlib.sha256((p/'joint_info.pkl').read_bytes()).hexdigest(),camera_sha256=hashlib.sha256((p/'ego_camera_info.pkl').read_bytes()).hexdigest(),points=points)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,enumerate(selected)))
(OUT/'manifest.json').write_text(json.dumps(rows,indent=2));pub=[{k:v for k,v in r.items() if k not in ['image','points']} for r in rows];(RUN/'evidence/preparation.json').write_text(json.dumps({'images':len(rows),'subjects':1,'actions':len(selected),'seed':20261118,'selection':'100 seeded shuffled action folders, middle available frame each; no selection by detector error','split':'convenience diagnostic; official heldout and upstream pretraining overlap not verified','reference':'native render joint_info.pkl, inverse per-frame Blender camera pose and [x,-y,-z], existing 01_12 calibration','calibration_sha256':hashlib.sha256(cal.read_bytes()).hexdigest(),'mapping':mapping,'samples':pub,'command':'prepare.py','exit_status':0},indent=2)+'\n');print('prepared',len(rows),flush=True)
