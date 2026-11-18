"""Select 50 test stereo pairs from readable local RGB and native JSON overlap."""
import os,json,tarfile,subprocess,hashlib,random,collections
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/cross_dataset100_20261118';DATA=Path(os.environ['UNREAL_ROOT'])
MAP=[('left_shoulder',5,'upperarm_l'),('right_shoulder',6,'upperarm_r'),('left_elbow',7,'lowerarm_l'),('right_elbow',8,'lowerarm_r'),('left_wrist',62,'hand_l'),('right_wrist',41,'hand_r'),('left_hip',9,'thigh_l'),('right_hip',10,'thigh_r'),('left_knee',11,'calf_l'),('right_knee',12,'calf_r'),('left_ankle',13,'foot_l'),('right_ankle',14,'foot_r')]
index=json.loads((OUT/'unreal_rgb_index.json').read_text());parts=sorted((DATA/'UnrealEgoData2_test_rgb').glob('CityPark_Showcase.tar.gz*'));pose=DATA/'UnrealEgoData2_test_pose/CityPark_Showcase.tar.gz000';pose_names=set()
with tarfile.open(pose,mode='r|gz') as t:
 for m in t:
  if m.isfile() and m.name.endswith('.json'):pose_names.add(m.name)
print('pose members',len(pose_names),flush=True)
pairs={}
for m in index['members']:
 name=m['member'];seq,tail=name.split('/fisheye_final_image/');side,file=tail.split('/');frame=int(Path(file).stem.split('_')[-1]);key=(seq,frame);pairs.setdefault(key,{})[side]=name
byseq=collections.defaultdict(list)
for (seq,frame),sides in pairs.items():
 pn=f'{seq}/json/frame_{frame}.json'
 if set(sides)=={'camera_left','camera_right'} and pn in pose_names:byseq[seq].append((seq,frame,sides,pn))
rng=random.Random(20261118);sequences=sorted(byseq);rng.shuffle(sequences);assert sum(map(len,byseq.values()))>=50
quotas=collections.Counter()
while sum(quotas.values())<50:
 for seq in sequences:
  if quotas[seq]<len(byseq[seq]):quotas[seq]+=1
  if sum(quotas.values())==50:break
selected=[]
for seq in sequences:
 pool=sorted(byseq[seq],key=lambda x:x[1]);n=quotas[seq]
 for i in range(n):selected.append(pool[rng.randrange(i*len(pool)//n,(i+1)*len(pool)//n)])
wanted_pose={p[3] for p in selected};annotations={}
with tarfile.open(pose,mode='r|gz') as t:
 for m in t:
  if m.name in wanted_pose:
   raw=t.extractfile(m).read();annotations[m.name]=(json.loads(raw),hashlib.sha256(raw).hexdigest())
assert len(annotations)==50
wanted={name:(i,side) for i,(_,_,sides,_) in enumerate(selected) for side,name in sides.items()};images={};image_dir=OUT/'unreal_images';image_dir.mkdir(exist_ok=True)
p=subprocess.Popen(['cat',*[str(x) for x in parts]],stdout=subprocess.PIPE);extract_error=None
try:
 with tarfile.open(fileobj=p.stdout,mode='r|gz') as t:
  for m in t:
   if m.name in wanted:
    raw=t.extractfile(m).read();assert len(raw)==m.size;im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);assert im is not None
    i,side=wanted[m.name];path=image_dir/f'pair_{i:03d}_{side}.png';path.write_bytes(raw);images[m.name]=(path,hashlib.sha256(raw).hexdigest(),im.shape)
    if len(images)%20==0:print('extracted images',len(images),flush=True)
   if len(images)==100:break
except Exception as e:extract_error=type(e).__name__+': '+str(e);raise
finally:p.stdout.close();cat_status=p.wait()
assert len(images)==100
rows=[]
for i,(seq,frame,sides,pn) in enumerate(selected):
 a,ah=annotations[pn];assert int(a['frame'])==frame
 for side in ['camera_left','camera_right']:
  image,ih,shape=images[sides[side]];h,w=shape[:2];assert (w,h)==(1024,1024);points=[]
  for name,sid,native in MAP:
   q=np.asarray(a['joints'][native][side+'_pts2d'],float);valid=bool(np.isfinite(q).all() and 0<=q[0]<w and 0<=q[1]<h)
   points.append(dict(name=name,sapiens_index=sid,native_name=native,reference=q.tolist(),reference_in_image=valid))
  rows.append(dict(dataset='unrealego2',sequence=seq,id=f'unrealego2_{len(rows):03d}',pair_id=f'ue2_pair_{i:03d}',frame=frame,side=side,image=str(image),image_sha256=ih,annotation_sha256=ah,source_image=sides[side],source_annotation=pn,width=w,height=h,points=points))
(OUT/'unrealego2_manifest.json').write_text(json.dumps(rows,indent=2))
for i in [0,25,50,75]:
 r=rows[i];im=cv2.imread(r['image'])
 for p in r['points']:
  if p['reference_in_image']:
   q=tuple(np.round(p['reference']).astype(int));cv2.circle(im,q,6,(0,255,255),2);cv2.putText(im,p['name'],q,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
 cv2.imwrite(str(OUT/f'unrealego2_overlay_{i}.jpg'),im)
ev={'seed':20261118,'images':100,'stereo_pairs':50,'scene':'CityPark_Showcase','subjects':sorted({r['sequence'].split('/')[2] for r in rows}),'sequence_pair_counts':dict(quotas),'available_complete_pairs_with_pose':sum(map(len,byseq.values())),'available_sequences':len(byseq),'rgb_archive_members_indexed':len(index['members']),'rgb_archive_scan_error':index['error'],'rgb_archive_scan_cat_exit':index['cat_exit'],'rgb_archive_parts':index['parts'],'extraction_cat_exit':cat_status,'extraction_cat_exit_note':'early stop once all 100 images found may close pipe and cause SIGPIPE','reference':'native camera_left/right_pts2d in released test pose JSON; original 1024x1024','selection':'balanced sequence quotas on paired RGB/pose intersection; seeded temporal strata; no detector-based selection','mapping':MAP,'visibility':'unknown; finite and in-frame only','samples':[{k:v for k,v in r.items() if k not in ['image','points']} for r in rows],'reference_in_image_points':sum(p['reference_in_image'] for r in rows for p in r['points']),'exit_status':0}
(RUN/'evidence/unrealego2_preparation.json').write_text(json.dumps(ev,indent=2)+'\n');print('DONE',len(rows),ev['subjects'],ev['reference_in_image_points'],flush=True)
