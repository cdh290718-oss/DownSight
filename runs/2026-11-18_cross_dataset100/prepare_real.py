"""Frozen balanced samples from released EgoPW/SceneEgo test annotations."""
import os,sys,pickle,json,random,hashlib
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/cross_dataset100_20261118'
sys.path.insert(0,str(ROOT.parent/'EgoSMPLX/main'))
from FishEyeCalibrated import FishEyeCameraCalibrated
MAP=[('left_shoulder',5,4),('right_shoulder',6,1),('left_elbow',7,5),('right_elbow',8,2),('left_wrist',62,6),('right_wrist',41,3),('left_hip',9,11),('right_hip',10,7),('left_knee',11,12),('right_knee',12,8),('left_ankle',13,13),('right_ankle',14,9)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def select(candidates,n,rng):
 assert len(candidates)>=n
 # One seeded sample per temporal stratum avoids adjacent-frame clusters.
 return [candidates[rng.randrange(int(i*len(candidates)/n),int((i+1)*len(candidates)/n))] for i in range(n)]
def record(dataset,seq,p,a,idx,uv,ann_hash):
 im=cv2.imread(str(p));assert im is not None;h,w=im.shape[:2];assert (w,h)==(1280,1024)
 points=[]
 for name,sid,j in MAP:
  q=np.asarray(uv[j],float);valid=bool(np.isfinite(q).all() and 0<=q[0]<w and 0<=q[1]<h)
  points.append(dict(name=name,sapiens_index=sid,native_index=j,reference=q.tolist(),reference_in_image=valid))
 return dict(dataset=dataset,sequence=seq,id=f'{dataset}_{idx:03d}',image=str(p),image_sha256=sha(p),annotation_sha256=ann_hash,annotation_image_name=a['image_name'],width=w,height=h,points=points)
for dataset in ['egopw','sceneego']:
 rng=random.Random(20261118);rows=[];stats=[];root=Path(os.environ[dataset.upper()+'_ROOT'])
 if dataset=='egopw':
  people=(root/'test_sequences.txt').read_text().split();groups=[]
  for person in people:
   seqs=sorted((root/person).glob('*/pseudo_gt.pkl'));groups.append((person,seqs))
  calib=None
 else:
  groups=[(s,[root/s/'annotation.pkl']) for s in ['jian1','jian2','diogo1','diogo2']]
  calib=ROOT.parent/'EgoSMPLX/main/fisheye.sceneego.calibration_05_08.json';camera=FishEyeCameraCalibrated(str(calib))
 for gi,(group,seqs) in enumerate(groups):
  quota=100//len(groups)+int(gi<100%len(groups))
  for si,ann in enumerate(seqs):
   n=quota//len(seqs)+int(si<quota%len(seqs))
   if not n:continue
   data=pickle.load(open(ann,'rb'));names={p.name for p in (ann.parent/'imgs').iterdir() if p.is_file()};available=[a for a in data if a['image_name'] in names];available.sort(key=lambda a:a['image_name'])
   stats.append({'sequence':str(ann.parent.relative_to(root)),'annotations':len(data),'available_rgb':len(available),'selected':n});ah=sha(ann)
   for a in select(available,n,rng):
    uv=a['ego_reprojected_pose'] if dataset=='egopw' else camera.world2camera(np.asarray(a['ego_pose_gt']))
    rows.append(record(dataset,str(ann.parent.relative_to(root)),ann.parent/'imgs'/a['image_name'],a,len(rows),uv,ah))
 assert len(rows)==100
 (OUT/(dataset+'_manifest.json')).write_text(json.dumps(rows,indent=2))
 # Four spread-out frames per dataset for reference projection inspection.
 for i in [0,25,50,75]:
  r=rows[i];im=cv2.imread(r['image'])
  for p in r['points']:
   if p['reference_in_image']:
    q=tuple(np.round(p['reference']).astype(int));cv2.circle(im,q,6,(0,255,255),2);cv2.putText(im,p['name'],q,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
  cv2.imwrite(str(OUT/f'{dataset}_overlay_{i}.jpg'),im)
 ev={'seed':20261118,'mapping':MAP,'images':100,'split':'released test list' if dataset=='egopw' else 'released test sequences jian1,jian2,diogo1,diogo2','reference':'released ego_reprojected_pose pseudo-labels' if dataset=='egopw' else 'released ego_pose_gt projected by existing SceneEgo calibration; no SMPL-X pseudo-GT','calibration_sha256':sha(calib) if calib else None,'selection':'balanced by test subject-directory/sequence; seeded temporal strata; no detector-dependent selection','stats':stats,'samples':[{k:v for k,v in r.items() if k not in ['image','points']} for r in rows],'reference_in_image_points':sum(p['reference_in_image'] for r in rows for p in r['points']),'visibility':'unknown, finite in-frame only','exit_status':0}
 (RUN/'evidence'/f'{dataset}_preparation.json').write_text(json.dumps(ev,indent=2)+'\n');print(dataset,len(rows),ev['reference_in_image_points'],flush=True)
